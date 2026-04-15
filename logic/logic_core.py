import requests
import datetime
import os
import json
import chromadb
from chromadb.utils import embedding_functions

# --- SEADISTUSED ---
OLLAMA_URL = "http://ollama:11434/api/generate"
OLLAMA_EMBED_URL = "http://ollama:11434/api/embeddings"
DB_PATH = "/app/storage/vector_db"
PROMPTS_FILE = "/app/logic/prompts.json"

def get_ee_time():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=3)))

def load_prompts():
    if os.path.exists(PROMPTS_FILE):
        try:
            with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"VIGA: Promptide laadimine ebaõnnestus: {e}")
    return {}

# Laadime promptid
PROMPTS = load_prompts()

def get_context(query, n_results=5):
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        ollama_ef = embedding_functions.OllamaEmbeddingFunction(
            url=OLLAMA_EMBED_URL,
            model_name="mxbai-embed-large"
        )
        collection = client.get_collection(name="procurements", embedding_function=ollama_ef)
        results = collection.query(query_texts=[query], n_results=n_results)
        return "\n\n".join(results['documents'][0])
    except Exception as e:
        return f"Konteksti leidmise viga: {str(e)}"

def ask_ollama(model, prompt, threads=4, timeout=360):
    try:
        payload = {
            "model": model, 
            "prompt": prompt, 
            "stream": False, 
            "keep_alive": -1, 
            "options": {"num_thread": threads}
        }
        response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        return response.json().get("response", "").strip() if response.status_code == 200 else f"VIGA_{response.status_code}"
    except Exception as e:
        return f"VIGA: {str(e)}"

def get_first_decision(text):
    if not text: return None
    t = text.upper()
    marks = [(t.find("ALLOWED"), "LUBATUD"), (t.find("LUBATUD"), "LUBATUD"),
             (t.find("BLOCKED"), "BLOKEERITUD"), (t.find("BLOKEERITUD"), "BLOKEERITUD")]
    found = sorted([m for m in marks if m[0] != -1])
    return found[0][1] if found else None