import os
import requests
import json
import re
from datetime import datetime
from zoneinfo import ZoneInfo
import chromadb
from chromadb.utils import embedding_functions

# --- KONFIGURATSIOON ---
OLLAMA_API_URL = "http://ollama:11434/api/generate"
CHROMA_DB_PATH = "/app/storage/vector_db"
PROMPTS_FILE = "/app/prompts.json"

# --- ANDMEBAASI SEADISTAMINE ---
embedding_func = embedding_functions.OllamaEmbeddingFunction(
    model_name="mxbai-embed-large",
    url="http://ollama:11434"
)

client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection(
    name="procurements", 
    embedding_function=embedding_func
)

# --- PROMPTIDE LAADIMINE ---
def load_prompts():
    """Laeb süsteemipromptid JSON failist."""
    try:
        if os.path.exists(PROMPTS_FILE):
            with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
    except Exception as e:
        print(f"VIGA PROMPTIDE LAADIMISEL: {e}")
        return {}

PROMPTS = load_prompts()

# --- ABIFUNKTSIOONID ---

def get_ee_time():
    """Tagastab Eesti ajaobjekti."""
    return datetime.now(ZoneInfo("Europe/Tallinn"))

def ask_ollama(model, prompt, threads, timeout):
    """Saadab päringu Ollama API-le."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_thread": threads,
            "num_predict": 1024,
            "temperature": 0
        }
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=timeout)
        if response.status_code == 200:
            return response.json().get("response", "")
        return f"VIGA: Ollama API status {response.status_code}"
    except Exception as e:
        return f"VIGA: {str(e)}"

def get_context(query, n_results=8):
    """Otsib VectorDB-st asjakohast konteksti."""
    try:
        results = collection.query(query_texts=[query], n_results=n_results)
        docs = results['documents'][0]
        metas = results.get('metadatas', [[]])[0]
        
        query_parts = re.findall(r'\w+', query.lower())
        scored_docs = []
        seen_content = set()
        
        for i, doc in enumerate(docs):
            snippet = doc[:200].strip().lower()
            if snippet in seen_content:
                continue
            seen_content.add(snippet)

            score = 0
            doc_lower = doc.lower()
            for part in query_parts:
                if len(part) > 5:
                    score += 2 * doc_lower.count(part)
                elif len(part) > 3:
                    score += doc_lower.count(part)
            
            source = metas[i].get("source", "Teadmata") if metas else "Teadmata"
            scored_docs.append((score, source, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # Fail fast: kui skoor on liiga madal, tagastame tühja stringi.
        # See annab main.py-le märku, et vastust pole mõtet genereerida.
        if not scored_docs or scored_docs[0][0] < 2:
            return ""

        context_parts = []
        for score, source, doc in scored_docs[:3]:
            context_parts.append(f"[ALLIKAS: {source}]\n{doc}")

        return "\n\n".join(context_parts)
    except Exception as e:
        print(f"VIGA KONTEKSTI OTSIMISEL: {e}")
        return ""

def parse_json_res(raw_res):
    """
    Turvaline JSON-i eraldamine ja parsimine tekstist ilma regex-ita.
    See meetod on immuunne süsteemi Markdowni tõrgetele.
    """
    try:
        clean_res = raw_res.strip()
        start_idx = clean_res.find('{')
        end_idx = clean_res.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            json_str = clean_res[start_idx:end_idx + 1]
            return json.loads(json_str)
        
        return json.loads(clean_res)
    except Exception:
        return {}

def parse_pre_check(raw_res):
    """Parsib eelkontrolli vastuse."""
    data = parse_json_res(raw_res)
    status = data.get("status", "BLOCKED")
    normalized = data.get("normalized_query", "")
    return status, normalized