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
MAX_CHUNK_CHARS = 800  # Turvaline piir bge-m3 ja mxbai jaoks

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
    """Salvestab impordi sündmused JSON logifaili."""
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

def get_clean_text(element):
    """Võtab elemendi vahetu teksti ilma sügavate järglasteta, et hoida struktuuri."""
    if element is None: return ""
    # Otsime tavateksti või sisuTeksti elemente
    tava = element.find(".//{*}tavatekst")
    if tava is not None and tava.text: return tava.text.strip()
    
    sisu = element.find(".//{*}sisuTekst")
    if sisu is not None:
        return "".join(sisu.itertext()).strip()
        
    return (element.text or "").strip()

def parse_xml_to_legal_chunks(file_path):
    """
    Kõrgema kvaliteediga parsimine: 
    - Eraldi metadata lõigete ja punktide jaoks.
    - Volitusnormi kaasamine teksti (embeddingu jaoks).
    - Max pikkuse kontroll.
    """
    chunks = []
    metadatas = []
    filename = os.path.basename(file_path)
    
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # 1. Dokumendi liik ja volitus
        doc_type_val = "muu dokument"
        doc_type_elem = root.find(".//{*}dokumentLiik")
        if doc_type_elem is not None and doc_type_elem.text:
            doc_type_val = doc_type_elem.text.strip().lower()
        
        volitus = ""
        v_elem = root.find(".//{*}volitusnorm") or root.find(".//{*}preambul")
        if v_elem is not None:
            volitus = re.sub(r'\s+', ' ', "".join(v_elem.itertext())).strip()
        
        # Volituse lühiversioon teksti sisse sulgudesse panemiseks
        volitus_context = f"({volitus[:100]}...)" if volitus else ""

        # 2. Seaduse lühend (garanteeritud)
        law_abbr = "AKT"
        lyhend_elem = root.find(".//{*}lyhend")
        if lyhend_elem is not None and lyhend_elem.text:
            law_abbr = lyhend_elem.text.strip()
        else:
            vja = root.find(".//{*}valjaandja")
            anr = root.find(".//{*}aktiNr")
            if vja is not None and anr is not None:
                law_abbr = f"{vja.text} {doc_type_val} nr {anr.text}"

        # 3. Parsime paragrahvid
        for para in root.findall(".//{*}paragrahv"):
            para_nr = (para.find("{*}paragrahvNr").text or "") if para.find("{*}paragrahvNr") is not None else ""
            
            def create_meta(display_prefix, lg="", p=""):
                m = {
                    "display_name": display_prefix,
                    "file": filename,
                    "section": str(para_nr),
                    "subsection": str(lg),
                    "point": str(p),
                    "type": doc_type_val
                }
                if volitus: m["volitus"] = volitus[:250]
                return m

            loiked = para.findall(".//{*}loige")
            
            if not loiked:
                content = get_clean_text(para)
                if content:
                    prefix = f"{law_abbr} § {para_nr}"
                    text = f"[{doc_type_val.upper()}] {prefix} {volitus_context}: {content}"[:MAX_CHUNK_CHARS]
                    chunks.append(text)
                    metadatas.append(create_meta(prefix))
            else:
                for lg in loiked:
                    lg_nr = (lg.find("{*}loigeNr").text or "") if lg.find("{*}loigeNr") is not None else ""
                    punktid = lg.findall(".//{*}alapunkt")
                    
                    if not punktid:
                        content = get_clean_text(lg)
                        if content:
                            prefix = f"{law_abbr} § {para_nr} lg {lg_nr}"
                            text = f"[{doc_type_val.upper()}] {prefix} {volitus_context}: {content}"[:MAX_CHUNK_CHARS]
                            chunks.append(text)
                            metadatas.append(create_meta(prefix, lg=lg_nr))
                    else:
                        # Intro tekst (punktide sissejuhatus)
                        intro = ""
                        for child in lg:
                            if strip_ns(child.tag) == "alapunkt": break
                            intro += "".join(child.itertext())
                        intro = intro.strip()

                        for p in punktid:
                            p_nr = (p.find("{*}alapunktNr").text or "") if p.find("{*}alapunktNr") is not None else ""
                            p_content = get_clean_text(p)
                            if p_content:
                                prefix = f"{law_abbr} § {para_nr} lg {lg_nr} p {p_nr}"
                                # Lisame intro ainult juhul kui see pole liiga pikk
                                full_content = f"{intro} {p_content}" if len(intro) < 200 else p_content
                                text = f"[{doc_type_val.upper()}] {prefix} {volitus_context}: {full_content}"[:MAX_CHUNK_CHARS]
                                chunks.append(text)
                                metadatas.append(create_meta(prefix, lg=lg_nr, p=p_nr))
                            
        return chunks, metadatas, doc_type_val
    except Exception as e:
        print(f"Viga {file_path} parsimisel: {e}")
        return [], [], "tundmatu"

def run_ingest():
    start_time = datetime.now()
    print(f"\n🚀 PRODUCTION-READY INGEST | Mudel: {EMBED_MODEL}")
    
    if not os.path.exists(LAWS_DIR):
        print(f"❌ VIGA: Kataloogi {LAWS_DIR} ei eksisteeri!")
        return

    files = [f for f in os.listdir(LAWS_DIR) if f.lower().endswith(".akt")]
    log_ingest_event("SYSTEM", "IMPORT_STARTED", {"files_found": len(files)})
    
    total_new_chunks = 0
    skipped_files = 0
    type_counts = {}

    for filename in files:
        # Kontrollime, kas fail on juba imporditud
        existing = collection.get(where={"file": filename}, limit=1)
        if existing and existing['ids']:
            print(f"⏩ {filename} on juba olemas.")
            skipped_files += 1
            continue

        path = os.path.join(LAWS_DIR, filename)
        chunks, metas, doc_type = parse_xml_to_legal_chunks(path)
        
        if chunks:
            print(f"📥 [{doc_type.upper()}] {filename} ({len(chunks)} osa)...", end=" ", flush=True)
            # Kasutame väiksemat batchi kindluse mõttes
            batch_size = 10
            for i in range(0, len(chunks), batch_size):
                end = i + batch_size
                collection.add(
                    documents=chunks[i:end],
                    ids=[f"id-{uuid.uuid4()}" for _ in chunks[i:end]],
                    metadatas=metas[i:end]
                )
            total_new_chunks += len(chunks)
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
            print(f"✅")
        else:
            print(f"⚠️ Failis {filename} puudus sisu.")

    duration = round((datetime.now() - start_time).total_seconds(), 2)
    log_ingest_event("SYSTEM", "IMPORT_FINISHED", {
        "new_chunks": total_new_chunks, 
        "duration_sec": duration,
        "types": type_counts
    })
    print(f"\n✨ IMPORT LÕPPES. Kestus: {duration}s. Kokku uusi osi: {total_new_chunks}")

if __name__ == "__main__":
    run_ingest()