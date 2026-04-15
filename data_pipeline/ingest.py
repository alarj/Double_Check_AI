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
    for filename in os.listdir(RAW_DIR):
        if filename.endswith(".xml"):
            path = os.path.join(RAW_DIR, filename)
            chunks = parse_procurement_xml(path)
            for i, chunk in enumerate(chunks):
                collection.add(
                    documents=[chunk["content"]],
                    metadatas=[chunk["metadata"]],
                    ids=[f"{filename}_{i}"]
                )
    print(f"Valmis! Andmed on nüüd vektorbaasis asukohas: {DB_PATH}")

if __name__ == "__main__":
    run()