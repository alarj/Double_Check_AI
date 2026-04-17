import os
import requests
import json
import re
from datetime import datetime
from zoneinfo import ZoneInfo
import chromadb
from chromadb.utils import embedding_functions

# --- KONFIGURATSIOON ---
# Võtame URL-id keskkonnamuutujatest (kooskõlas docker-compose'iga)
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_API_URL = f"{OLLAMA_URL}/api/generate"
# Täpne path vastavalt docker-compose volume'ile
CHROMA_DB_PATH = "/app/storage/vector_db"
PROMPTS_FILE = "/app/prompts.json"

# --- ANDMEBAASI SEADISTAMINE ---
# KRIITILINE: Sünkroonis sinu viimase 'bge-m3' ingestiga
EMBEDDING_MODEL = "bge-m3"

embedding_func = embedding_functions.OllamaEmbeddingFunction(
    model_name=EMBEDDING_MODEL,
    url=OLLAMA_URL
)

# PersistentClient tagab ligipääsu salvestatud andmetele
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection(
    name="procurements", 
    embedding_function=embedding_func
)

# --- PROMPTIDE LAADIMINE ---
def load_prompts():
    """Laeb süsteemi juhised ja mallid JSON failist."""
    try:
        if os.path.exists(PROMPTS_FILE):
            with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
    except Exception as e:
        print(f"VIGA PROMPTIDE LAADIMISEL: {e}")
        return {}

# Globaalne promptide objekt, mida main.py saab reaalajas kasutada
PROMPTS = load_prompts()

# --- ABI-FUNKTSIOONID ---

def get_ee_time():
    """Tagastab Eesti aja logimiseks."""
    return datetime.now(ZoneInfo("Europe/Tallinn"))

def get_context(query, n_results=5):
    """
    Teostab RAG-otsingu koos hübriidse skoorimisega (Vektor + Märksõnad).
    Optimeeritud bge-m3 distantsidele.
    """
    try:
        # Küsime vektorbaasist kandidaadid
        results = collection.query(query_texts=[query], n_results=n_results)
        
        if not results or not results['documents'] or not results['documents'][0]:
            return ""

        docs = results['documents'][0]
        metas = results.get('metadatas', [[]])[0]
        distances = results.get('distances', [[]])[0]
        
        query_words = re.findall(r'\w+', query.lower())
        scored_docs = []
        seen_snippets = set()
        
        for i, doc in enumerate(docs):
            # Dublikaatide eemaldamine sisu alguse põhjal
            snippet = doc[:100].strip().lower()
            if snippet in seen_snippets: continue
            seen_snippets.add(snippet)

            # Skoorimine bge-m3 jaoks:
            # L2 distants: 0 on identne. 1.4 on piirväärtus, millest edasi on seos nõrk.
            v_score = max(0, 1.4 - distances[i])
            
            # Märksõnade täiendav kaal (Hübriidne otsing)
            k_score = 0
            doc_lower = doc.lower()
            for word in query_words:
                # Anname kaalu ainult sisulistele sõnadele
                if len(word) > 4 and word in doc_lower:
                    k_score += 0.4
            
            final_score = v_score + k_score
            source = metas[i].get("source", "RHS") if metas else "RHS"
            scored_docs.append((final_score, source, doc))

        # Sorteerime tulemused lõpliku hübriidse skoori järgi
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # FAIL FAST lävend: bge-m3 puhul on 0.65-0.7 turvaline piir
        # See hoiab ära hallutsinatsioonid tühja konteksti baasilt
        if not scored_docs or scored_docs[0][0] < 0.65:
            return ""

        formatted_results = []
        for sc, s, d in scored_docs[:3]:
            formatted_results.append(f"--- ALLIKAS: {s} ---\n{d}")
            
        return "\n\n".join(formatted_results)
        
    except Exception as e:
        print(f"VIGA konteksti loomisel: {e}")
        return ""

def ask_ollama(model, prompt, threads, timeout):
    """Saadab päringu Ollama API-le genereerimiseks."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_thread": int(threads),
            "num_predict": 1024,
            "temperature": 0  # Deterministlik vastus on juriidikas kriitiline
        }
    }
    
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=timeout)
        if response.status_code == 200:
            return response.json().get("response", "")
        return f"VIGA: Ollama vastas koodiga {response.status_code}"
    except requests.exceptions.Timeout:
        return f"VIGA: Aegumine ({timeout}s)."
    except Exception as e:
        return f"VIGA: Sidekatkestus - {str(e)}"

def parse_json_res(raw_res):
    """
    Robustne JSON parsimine tekstist.
    Eemaldab võimalikud Markdowni koodiplokid ja prügi ümber JSON-i.
    """
    if not raw_res: 
        return {}
        
    try:
        text = str(raw_res).strip()
        
        # Otsime üles esimese '{' ja viimase '}'
        start = text.find('{')
        end = text.rfind('}') + 1
        
        if start != -1 and end > 0:
            json_str = text[start:end]
            return json.loads(json_str)
        
        # Kui sulge ei leitud, proovime parsida tervet teksti
        return json.loads(text)
    except Exception:
        return {}

def parse_pre_check(raw_res):
    """Eraldab eelkontrolli staatuse ja normaliseeritud päringu."""
    data = parse_json_res(raw_res)
    return data.get("status", "BLOCKED"), data.get("normalized_query", "")