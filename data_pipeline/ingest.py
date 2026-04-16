import os
import chromadb
from chromadb.utils import embedding_functions
from parsers import parse_procurement_xml

# Seadistus - liigume ühe tase üles ja siis storage kausta
DB_PATH = os.path.abspath("../storage/vector_db/")
RAW_DIR = os.path.abspath("../storage/raw/procurements/")

# Ollama embeddingud
ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name="mxbai-embed-large"
)

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(name="procurements", embedding_function=ollama_ef)

def run():
    print(f"--- Alustan hangete sisselugemist asukohast: {RAW_DIR} ---")
    total_new = 0
    total_skipped = 0
    
    # Sorteerime failid, et nimekiri jookseks ekraanil loogiliselt
    files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith(".xml")])
    
    for filename in files:
        # Kuvame kohe töötlemisel oleva faili nime (ilma reavahetuseta veel)
        print(f"Töötlen: {filename}...", end=" ", flush=True)
        
        # KONTROLL: Kas selle faili andmed on juba baasis?
        check_id = f"{filename}_0"
        existing = collection.get(ids=[check_id])
        
        if existing and existing['ids']:
            total_skipped += 1
            print("Hüppan üle (baasis olemas) ✅")
            continue

        path = os.path.join(RAW_DIR, filename)
        try:
            chunks = parse_procurement_xml(path)
            if chunks:
                for i, chunk in enumerate(chunks):
                    collection.add(
                        documents=[chunk["content"]],
                        metadatas=[chunk["metadata"]],
                        ids=[f"{filename}_{i}"]
                    )
                total_new += 1
                print(f"✅ Lisatud {len(chunks)} osa.")
            else:
                print("⚠️ Sisu ei leitud.")
                
        except Exception as e:
            print(f"❌ Viga: {e}")

    print(f"\n--- TÖÖ LÕPP ---")
    print(f"Uusi hankeid lisatud: {total_new}")
    print(f"Vahele jäetud (juba olemas): {total_skipped}")
    print(f"Andmete asukoht: {DB_PATH}")

if __name__ == "__main__":
    run()