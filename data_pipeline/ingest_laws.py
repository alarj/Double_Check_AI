import os
import lxml.etree as ET
import chromadb
from chromadb.utils import embedding_functions

# Seadistused
LAWS_DIR = "../storage/raw/laws"
DB_PATH = "../storage/vector_db"
OLLAMA_URL = "http://localhost:11434/api/embeddings"

def parse_law_xml(filepath):
    """Tükeldab Riigi Teataja XML-i paragrahvide kaupa."""
    tree = ET.parse(filepath)
    root = tree.getroot()
    
    # Leiame seaduse pealkirja
    law_title = root.find(".//aktPealkiri/pealkiri/tekst").text
    
    chunks = []
    # Otsime kõik paragrahvid (§)
    for para in root.xpath(".//paragrahv"):
        para_id = para.get("id", "tundmatu")
        # Kogume kokku paragrahvi pealkirja ja sisu
        para_text = "".join(para.itertext()).strip()
        
        if para_text:
            chunks.append({
                "text": f"{law_title}: {para_text}",
                "metadata": {
                    "source": os.path.basename(filepath),
                    "law_title": law_title,
                    "para_id": para_id,
                    "type": "law"
                }
            })
    return chunks

def ingest_laws():
    client = chromadb.PersistentClient(path=DB_PATH)
    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        url=OLLAMA_URL,
        model_name="mxbai-embed-large"
    )
    collection = client.get_or_create_collection(name="procurements", embedding_function=ollama_ef)

    for filename in os.listdir(LAWS_DIR):
        if filename.endswith(".akt") or filename.endswith(".xml"):
            path = os.path.join(LAWS_DIR, filename)
            print(f"Töötlen seadust: {filename}...")
            
            chunks = parse_law_xml(path)
            
            for i, chunk in enumerate(chunks):
                collection.add(
                    documents=[chunk["text"]],
                    metadatas=[chunk["metadata"]],
                    ids=[f"law_{filename}_{i}"]
                )
    print("Kõik seadused on vektorbaasi lisatud!")

if __name__ == "__main__":
    ingest_laws()