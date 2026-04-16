import os
import xml.etree.ElementTree as ET
import uuid
import chromadb
import time
import re
from chromadb.utils import embedding_functions

# --- SEADISTUSED ---
# Kasutame pildil näidatud ja kirjeldatud kataloogistruktuuri
LAWS_DIR = "/app/storage/raw/laws/"
DB_PATH = "/app/storage/vector_db"
OLLAMA_URL = "http://ollama:11434" # Docker-vahelise suhtluse aadress kui jookseb dockeris
# OLLAMA_URL = "http://localhost:11434" # benchmarkimise jaoks otse serveris

# --- ANDMEBAASI ÜHENDUS ---
embedding_func = embedding_functions.OllamaEmbeddingFunction(
    model_name="mxbai-embed-large",
    url=OLLAMA_URL
)

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(
    name="procurements", 
    embedding_function=embedding_func
)

def clean_text(text):
    """Puhastab teksti liigsetest tühikutest."""
    if not text: return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def parse_law_xml(file_path):
    """
    Parsib .akt faili ja eraldab paragrahvid koos metaandmetega.
    Struktuur: <paragrahv> -> <paragrahvNr>, <kuvatavTekst>, <sisu>
    """
    chunks = []
    metadatas = []
    
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Leime lühendi (nt RHS)
        lyhend = "SEADUS"
        meta = root.find(".//{tyviseadus_1_10.02.2010}metaandmed")
        if meta is not None:
            l_elem = meta.find("{tyviseadus_1_10.02.2010}lyhend")
            if l_elem is not None: lyhend = l_elem.text

        # Otsime kõik paragrahvid üle kogu dokumendi
        # Märkus: namespace võib varieeruda, seetõttu kasutame '//*' lähenemist teatud elementide puhul
        for para in root.iter():
            if para.tag.endswith('paragrahv'):
                para_nr = ""
                para_pealkiri = ""
                para_sisu = []

                # Paragrahvi number ja pealkiri
                nr_elem = para.find(".//*[@id]") # Sageli on ID-ga elemendid numbrid
                for child in para:
                    tag = child.tag.split('}')[-1]
                    if tag == "paragrahvNr":
                        para_nr = child.text
                    elif tag == "kuvatavTekst":
                        para_pealkiri = child.text
                    elif tag == "sisu":
                        # Kogume kogu teksti sisu elemendi alt (lõiked, punktid)
                        para_sisu.append(" ".join(child.itertext()))

                full_content = clean_text(" ".join(para_sisu))
                if full_content:
                    # Loome tervikliku teksti: "§ 1. Pealkiri. Sisu..."
                    display_header = f"§ {para_nr}. {para_pealkiri}".strip(". ")
                    document_text = f"{lyhend} {display_header}: {full_content}"
                    
                    chunks.append(document_text)
                    metadatas.append({
                        "source": os.path.basename(file_path),
                        "type": "law",
                        "section": para_nr,
                        "law_name": lyhend
                    })
                    
        return chunks, metadatas
    except Exception as e:
        print(f"Viga XML parsimisel {file_path}: {e}")
        return [], []

def run_ingest():
    print(f"--- ALUSTAN SEADUSTE IMPORTI ({LAWS_DIR}) ---")
    total_added = 0
    
    if not os.path.exists(LAWS_DIR):
        print(f"VIGA: Kataloogi {LAWS_DIR} ei leitud!")
        return

    files = [f for f in os.listdir(LAWS_DIR) if f.endswith(".akt")]
    
    for filename in files:
        # Kontrollime duplikaate (allika põhjal)
        existing = collection.get(where={"source": filename}, limit=1)
        if existing and existing['ids']:
            print(f"Hüppan üle: {filename} (juba olemas) ✅")
            continue

        path = os.path.join(LAWS_DIR, filename)
        print(f"Töötlen: {filename}...", end=" ", flush=True)
        
        chunks, metas = parse_law_xml(path)
        
        if chunks:
            # Lisame andmed baasi väikeste pakkidena (batch), et vältida timeout-e
            batch_size = 5
            for i in range(0, len(chunks), batch_size):
                end = i + batch_size
                collection.add(
                    documents=chunks[i:end],
                    ids=[str(uuid.uuid4()) for _ in chunks[i:end]],
                    metadatas=metas[i:end]
                )
                time.sleep(0.05) # Väike hingetõmbeaeg API-le
            
            total_added += len(chunks)
            print(f"✅ Lisatud {len(chunks)} paragrahvi.")
        else:
            print(f"⚠️ Paragrahve ei leitud.")

    print(f"--- IMPORT LÕPETATUD. Kokku lisati {total_added} kirjet. ---")

if __name__ == "__main__":
    run_ingest()