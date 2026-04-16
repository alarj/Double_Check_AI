import os
import xml.etree.ElementTree as ET
import uuid
import chromadb
import re
import json
from datetime import datetime
from chromadb.utils import embedding_functions

# --- KONFIGURATSIOON ---
LAWS_DIR = "/app/storage/raw/laws/"
DB_PATH = "/app/storage/vector_db"
INGEST_LOG_FILE = "/app/storage/ingest.log"
OLLAMA_URL = "http://ollama:11434"
EMBED_MODEL = "bge-m3"

# --- ANDMEBAASI ÜHENDUS ---
embedding_func = embedding_functions.OllamaEmbeddingFunction(
    model_name=EMBED_MODEL,
    url=OLLAMA_URL
)

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(
    name="procurements", 
    embedding_function=embedding_func
)

def log_ingest_event(doc_type, event_name, details):
    """Salvestab impordi sündmused eraldi logifaili."""
    try:
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "doc_type": doc_type,
            "event": event_name,
            "details": details
        }
        with open(INGEST_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"VIGA LOGIMISEL: {e}")

def strip_ns(tag):
    return tag.split('}')[-1] if '}' in tag else tag

def parse_xml_to_legal_chunks(file_path):
    """
    Parsib XML-i, tuvastab liigi ja otsib volitusinfot (seost seadusega) 
    nii volitusnormi märgendist kui ka preambulast.
    """
    chunks = []
    metadatas = []
    filename = os.path.basename(file_path)
    
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # 1. Dokumendi liik (seadus, määrus jne)
        doc_type_val = "muu dokument"
        doc_type_elem = root.find(".//{*}dokumentLiik")
        if doc_type_elem is not None and doc_type_elem.text:
            doc_type_val = doc_type_elem.text.strip().lower()
        
        doc_type_display = doc_type_val.upper()

        # 2. Volitusinfo otsimine (Määruste puhul oluline seos seadusega)
        volitus = ""
        # Variant A: Spetsiaalne märgend
        volitus_elem = root.find(".//{*}volitusnorm")
        if volitus_elem is not None:
            volitus = "".join(volitus_elem.itertext()).strip()
        
        # Variant B: Preambul (Sinu failis 115082017003.akt on info siin)
        if not volitus:
            preambul_elem = root.find(".//{*}preambul")
            if preambul_elem is not None:
                volitus = "".join(preambul_elem.itertext()).strip()
                # Puhastame liigse tühja ja reavahetused
                volitus = re.sub(r'\s+', ' ', volitus).strip()

        # 3. Viide/lühend (nt RHS)
        law_abbr = ""
        lyhend_elem = root.find(".//{*}lyhend")
        if lyhend_elem is not None and lyhend_elem.text:
            law_abbr = lyhend_elem.text
        
        if not law_abbr:
            valjaandja = root.find(".//{*}valjaandja")
            akti_nr = root.find(".//{*}aktiNr")
            if valjaandja is not None and akti_nr is not None:
                law_abbr = f"{valjaandja.text} {doc_type_val} nr {akti_nr.text}"
            else:
                law_abbr = doc_type_val.capitalize()

        # 4. Parsime sisu paragrahvi tasemel
        for para in root.findall(".//{*}paragrahv"):
            para_nr = ""
            para_nr_elem = para.find("{*}paragrahvNr")
            if para_nr_elem is not None:
                para_nr = para_nr_elem.text
            
            def create_meta(prefix):
                m = {
                    "source": prefix, 
                    "file": filename, 
                    "section": str(para_nr),
                    "type": doc_type_val
                }
                if volitus: m["volitus"] = volitus[:250] # Salvestame seose alusega
                return m

            loiked = para.findall(".//{*}loige")
            if not loiked:
                content = "".join(para.itertext()).strip()
                if content:
                    prefix = f"{law_abbr} § {para_nr}"
                    chunks.append(f"[{doc_type_display}] {prefix}: {content}")
                    metadatas.append(create_meta(prefix))
            else:
                for lg in loiked:
                    lg_nr = ""
                    lg_nr_elem = lg.find("{*}loigeNr")
                    if lg_nr_elem is not None:
                        lg_nr = lg_nr_elem.text
                    
                    punktid = lg.findall(".//{*}alapunkt")
                    if not punktid:
                        content = "".join(lg.itertext()).strip()
                        if content:
                            prefix = f"{law_abbr} § {para_nr} lg {lg_nr}"
                            chunks.append(f"[{doc_type_display}] {prefix}: {content}")
                            metadatas.append(create_meta(prefix))
                    else:
                        intro = ""
                        for child in lg:
                            if strip_ns(child.tag) == "alapunkt": break
                            intro += "".join(child.itertext())
                        
                        for p in punktid:
                            p_nr = ""
                            p_nr_elem = p.find("{*}alapunktNr")
                            if p_nr_elem is not None: p_nr = p_nr_elem.text
                            p_content = "".join(p.itertext()).strip()
                            prefix = f"{law_abbr} § {para_nr} lg {lg_nr} p {p_nr}"
                            chunks.append(f"[{doc_type_display}] {prefix}: {intro.strip()} {p_content}")
                            metadatas.append(create_meta(prefix))
                            
        return chunks, metadatas, doc_type_val
    except Exception as e:
        print(f"Viga {file_path} parsimisel: {e}")
        return [], [], "tundmatu"

def run_ingest():
    start_time = datetime.now()
    print(f"\n🚀 STRUKTUURNE IMPORT | DOKUMENDI LIIGID JA VOLITUSED")
    
    if not os.path.exists(LAWS_DIR):
        print(f"❌ VIGA: Kataloogi {LAWS_DIR} ei eksisteeri!")
        return

    files = [f for f in os.listdir(LAWS_DIR) if f.lower().endswith(".akt")]
    log_ingest_event("SYSTEM", "IMPORT_STARTED", {"files_found": len(files)})
    
    total_new_chunks = 0
    skipped_files = 0
    type_counts = {}

    for filename in files:
        existing = collection.get(where={"file": filename}, limit=1)
        if existing and existing['ids']:
            print(f"⏩ {filename} on juba olemas.")
            skipped_files += 1
            continue

        path = os.path.join(LAWS_DIR, filename)
        chunks, metas, doc_type = parse_xml_to_legal_chunks(path)
        
        if chunks:
            print(f"📥 [{doc_type.upper()}] {filename}...", end=" ", flush=True)
            batch_size = 20
            for i in range(0, len(chunks), batch_size):
                end = i + batch_size
                collection.add(
                    documents=chunks[i:end],
                    ids=[f"doc-{uuid.uuid4()}" for _ in chunks[i:end]],
                    metadatas=metas[i:end]
                )
            total_new_chunks += len(chunks)
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
            print(f"✅ {len(chunks)} osa lisatud.")
        else:
            print(f"⚠️ Failis {filename} puudus struktuurne sisu.")

    duration = (datetime.now() - start_time).total_seconds()
    log_ingest_event("SYSTEM", "IMPORT_FINISHED", {
        "new_chunks": total_new_chunks, 
        "skipped_files": skipped_files,
        "types_summary": type_counts,
        "duration_sec": round(duration, 2)
    })
    
    print(f"\n✨ VALMIS! Statistika: {type_counts}")

if __name__ == "__main__":
    run_ingest()