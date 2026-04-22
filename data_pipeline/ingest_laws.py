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
MAX_INTRO_CHARS = 250
MIN_CONTENT_WITHOUT_INTRO = 120

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

def normalize_text(text):
    """Normaliseerib tühikud ja hoiab kirjavahemärgid loetavad."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text

def extract_text_with_spacing(element):
    """Kogub elemendi teksti nii, et sõnade ja loetelude vahele jääks eraldus."""
    if element is None:
        return ""
    parts = [part.strip() for part in element.itertext() if part and part.strip()]
    return normalize_text(" ".join(parts))

def get_numbered_marker(element, number_tag):
    number_elem = element.find(number_tag)
    if number_elem is not None:
        number = extract_text_with_spacing(number_elem)
        if number:
            return f"{number})"
    return "-"

def serialize_legal_structure(element):
    """Hoiab otseste juriidiliste alamstruktuuride markerid tekstis alles."""
    if element is None:
        return ""

    structured_parts = []
    direct_children = list(element)

    for child in direct_children:
        tag = strip_ns(child.tag)

        if tag in {"tavatekst", "sisuTekst", "kuvatavTekst"}:
            child_text = extract_text_with_spacing(child)
            if child_text:
                structured_parts.append(child_text)
            continue

        if tag == "alapunkt":
            marker = get_numbered_marker(child, "{*}alapunktNr")
            child_text = serialize_legal_structure(child)
            if child_text:
                structured_parts.append(f"{marker} {child_text}")
            continue

        if tag == "loige":
            marker = get_numbered_marker(child, "{*}loigeNr")
            child_text = serialize_legal_structure(child)
            if child_text:
                structured_parts.append(f"{marker} {child_text}")
            continue

    if structured_parts:
        return normalize_text(" ".join(structured_parts))

    return extract_text_with_spacing(element)

def smart_truncate(text, max_chars):
    """Kärbib teksti võimalusel lause või fraasi piirilt, mitte suvalisest kohast."""
    text = normalize_text(text)
    if len(text) <= max_chars:
        return text

    cutoff = text[:max_chars + 1]
    preferred_breaks = [cutoff.rfind(mark) for mark in [". ", "; ", ": ", "! ", "? "]]
    preferred_break = max(preferred_breaks)
    if preferred_break >= int(max_chars * 0.7):
        return cutoff[:preferred_break + 1].rstrip()

    word_break = cutoff.rfind(" ")
    if word_break >= int(max_chars * 0.7):
        return cutoff[:word_break].rstrip() + "..."

    return cutoff[:max_chars].rstrip() + "..."

def get_paragraph_title(para):
    title_elem = para.find("{*}paragrahvPealkiri") or para.find("{*}pealkiri")
    return extract_text_with_spacing(title_elem)

def build_chunk_text(doc_type, prefix, content, para_title=""):
    title_suffix = f" ({para_title})" if para_title else ""
    return smart_truncate(
        f"[{doc_type.upper()}] {prefix}{title_suffix}: {content}",
        MAX_CHUNK_CHARS
    )

def get_clean_text(element):
    """Võtab elemendi teksti nii, et loetelud ja sisemised eraldused jääks alles."""
    if element is None:
        return ""
    return serialize_legal_structure(element)

def parse_xml_to_legal_chunks(file_path):
    """
    Kõrgema kvaliteediga parsimine: 
    - Eraldi metadata lõigete ja punktide jaoks.
    - Paragrahvi pealkirja lisamine embeddingu teksti.
    - Volitusnorm metadata-s, mitte embeddingu tekstis.
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
            para_title = get_paragraph_title(para)
            
            def create_meta(display_prefix, chunk_type, lg="", p=""):
                m = {
                    "law": law_abbr,
                    "display_name": display_prefix,
                    "file": filename,
                    "section": str(para_nr),
                    "subsection": str(lg),
                    "point": str(p),
                    "type": doc_type_val,
                    "chunk_type": chunk_type
                }
                if para_title:
                    m["paragraph_title"] = para_title[:250]
                if volitus:
                    m["volitus"] = volitus[:250]
                return m

            loiked = para.findall(".//{*}loige")
            
            if not loiked:
                content = get_clean_text(para)
                if content:
                    prefix = f"{law_abbr} § {para_nr}"
                    text = build_chunk_text(doc_type_val, prefix, content, para_title)
                    chunks.append(text)
                    metadatas.append(create_meta(prefix, "section"))
            else:
                for lg in loiked:
                    lg_nr = (lg.find("{*}loigeNr").text or "") if lg.find("{*}loigeNr") is not None else ""
                    punktid = lg.findall(".//{*}alapunkt")
                    
                    if not punktid:
                        content = get_clean_text(lg)
                        if content:
                            prefix = f"{law_abbr} § {para_nr} lg {lg_nr}"
                            text = build_chunk_text(doc_type_val, prefix, content, para_title)
                            chunks.append(text)
                            metadatas.append(create_meta(prefix, "subsection", lg=lg_nr))
                    else:
                        # Intro tekst (punktide sissejuhatus)
                        intro_parts = []
                        for child in lg:
                            if strip_ns(child.tag) == "alapunkt":
                                break
                            child_text = extract_text_with_spacing(child)
                            if child_text:
                                intro_parts.append(child_text)
                        intro = normalize_text(" ".join(intro_parts))

                        lg_prefix = f"{law_abbr} § {para_nr} lg {lg_nr}"
                        lg_content = get_clean_text(lg)
                        if lg_content:
                            lg_text = build_chunk_text(doc_type_val, lg_prefix, lg_content, para_title)
                            chunks.append(lg_text)
                            metadatas.append(create_meta(lg_prefix, "subsection", lg=lg_nr))

                        for p in punktid:
                            p_nr = (p.find("{*}alapunktNr").text or "") if p.find("{*}alapunktNr") is not None else ""
                            p_content = get_clean_text(p)
                            if p_content:
                                prefix = f"{law_abbr} § {para_nr} lg {lg_nr} p {p_nr}"
                                needs_intro = len(p_content) < MIN_CONTENT_WITHOUT_INTRO
                                should_attach_intro = intro and (len(intro) <= MAX_INTRO_CHARS or needs_intro)
                                full_content = normalize_text(f"{intro} {p_content}") if should_attach_intro else p_content
                                text = build_chunk_text(doc_type_val, prefix, full_content, para_title)
                                chunks.append(text)
                                metadatas.append(create_meta(prefix, "point", lg=lg_nr, p=p_nr))
                            
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
            
            # Logime faili eduka impordi
            log_ingest_event(doc_type, "FILE_IMPORTED", {
                "file": filename, 
                "chunks": len(chunks)
            })
            print(f"✅")
        else:
            print(f"⚠️ Failis {filename} puudus sisu.")

    duration = round((datetime.now() - start_time).total_seconds(), 2)
    log_ingest_event("SYSTEM", "IMPORT_FINISHED", {
        "new_chunks": total_new_chunks, 
        "duration_sec": duration,
        "types": type_counts,
        "skipped": skipped_files
    })
    print(f"\n✨ IMPORT LÕPPES. Kestus: {duration}s. Kokku uusi osi: {total_new_chunks}")

if __name__ == "__main__":
    run_ingest()
