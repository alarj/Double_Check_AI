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
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
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
            # utf-8-sig loeb korrektselt ka BOM-iga JSON failid
            # (nt kui fail on salvestatud Windowsi tööriistadega).
            with open(PROMPTS_FILE, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
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

def _first_result_list(results, key):
    """Tagastab Chroma query tulemuse esimese listi turvaliselt."""
    if not isinstance(results, dict):
        return []
    value = results.get(key)
    if not value:
        return []
    if isinstance(value, list):
        if not value:
            return []
        first = value[0]
        return first if isinstance(first, list) else []
    return []

def get_context(query, n_results=5, max_context_blocks=3, return_debug=False):
    """
    Teostab RAG-otsingu koos hübriidse skoorimisega (Vektor + Märksõnad).
    Optimeeritud bge-m3 distantsidele.
    """
    try:
        # Küsime vektorbaasist rohkem kandidaate kui lõppväljundisse vaja,
        # et Pythonis tehtav ümberreastamine ja dedupe saaks päriselt mõjuda.
        fetch_k = max(int(n_results or 5), int(max_context_blocks or 3), 5) * 4
        results = collection.query(query_texts=[query], n_results=fetch_k)
        
        docs = _first_result_list(results, "documents")
        if not docs:
            if return_debug:
                return "", {"fetch_k": fetch_k, "candidates": []}
            return ""

        metas = _first_result_list(results, "metadatas")
        distances = _first_result_list(results, "distances")
        if len(metas) < len(docs):
            metas = metas + [{} for _ in range(len(docs) - len(metas))]
        if len(distances) < len(docs):
            distances = distances + [1.4 for _ in range(len(docs) - len(distances))]
        
        query_words = re.findall(r'\w+', query.lower())
        query_numbers = re.findall(r'\d[\d\s]*', query.lower())
        query_stems = {word[:5] for word in query_words if len(word) >= 6}
        scored_docs = []
        seen_snippets = set()
        
        for i, doc in enumerate(docs):
            if not isinstance(doc, str) or not doc.strip():
                continue
            # Dublikaatide eemaldamine sisu alguse põhjal
            snippet = doc[:100].strip().lower()
            if snippet in seen_snippets:
                continue
            seen_snippets.add(snippet)

            # Skoorimine bge-m3 jaoks:
            # L2 distants: 0 on identne. 1.4 on piirväärtus, millest edasi on seos nõrk.
            v_score = max(0, 1.4 - distances[i])
            
            # Märksõnade täiendav kaal (Hübriidne otsing)
            k_score = 0
            doc_lower = doc.lower()
            meta = metas[i] if i < len(metas) and isinstance(metas[i], dict) else {}
            display_name = str(meta.get("display_name", "")).lower()
            para_title = str(meta.get("paragraph_title", "")).lower()
            chunk_type = str(meta.get("chunk_type", "")).lower()
            title_text = f"{display_name} {para_title}".strip()
            title_words = set(re.findall(r'\w+', title_text))
            title_stems = {word[:5] for word in title_words if len(word) >= 6}
            doc_words = set(re.findall(r'\w+', doc_lower))
            doc_stems = {word[:5] for word in doc_words if len(word) >= 6}

            for word in query_words:
                # Anname kaalu ainult sisulistele sõnadele
                if len(word) > 4 and word in doc_lower:
                    k_score += 0.4
                if len(word) > 4 and word in para_title:
                    k_score += 0.45
                if len(word) > 2 and word in display_name:
                    k_score += 0.25

            for number in query_numbers:
                normalized_number = re.sub(r"\s+", "", number)
                if normalized_number and normalized_number in re.sub(r"\s+", "", doc_lower):
                    k_score += 0.6

            shared_title_stems = query_stems & title_stems
            shared_doc_stems = query_stems & doc_stems
            k_score += 0.18 * len(shared_doc_stems)
            k_score += 0.3 * len(shared_title_stems)

            if query.lower() in doc_lower:
                k_score += 0.5
            if query.lower() in para_title:
                k_score += 0.4
            if query.lower() in display_name:
                k_score += 0.25

            # Kui enamik päringu sisulistest sõnadest elab pealkirjas,
            # tasub see tõsta kõrgemale ka siis, kui vektorotsing eelistab detailseid erandeid.
            meaningful_words = {word for word in query_words if len(word) > 4}
            if meaningful_words:
                title_overlap = meaningful_words & title_words
                if len(title_overlap) >= max(1, len(meaningful_words) - 1):
                    k_score += 0.7

            # Üldiste teemaküsimuste puhul eelistame section/subsection taset punktidele.
            if len(query_words) <= 5 and chunk_type == "section":
                k_score += 0.2

            if chunk_type == "subsection":
                k_score += 0.3
            elif chunk_type == "section":
                k_score += 0.15
            elif chunk_type == "point":
                k_score -= 0.05
            
            final_score = v_score + k_score
            source = meta.get("source") or meta.get("law") or "RHS"
            family_key = (
                str(meta.get("law", "")),
                str(meta.get("section", "")),
                str(meta.get("subsection", "")),
            )
            scored_docs.append((final_score, source, doc, family_key, meta))

        # Sorteerime tulemused lõpliku hübriidse skoori järgi
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # FAIL FAST lävend: bge-m3 puhul on 0.65-0.7 turvaline piir
        # See hoiab ära hallutsinatsioonid tühja konteksti baasilt
        if not scored_docs or scored_docs[0][0] < 0.65:
            if return_debug:
                return "", {
                    "fetch_k": fetch_k,
                    "candidates": [
                        {
                            "rank": i + 1,
                            "score": round(sc, 4),
                            "source": s,
                            "selected": False,
                            "metadata": m,
                            "text": d,
                        }
                        for i, (sc, s, d, _family_key, m) in enumerate(scored_docs)
                    ],
                }
            return ""

        formatted_results = []
        seen_families = set()
        selected_indexes = set()
        limit = max(1, int(max_context_blocks or 3))
        for selected_index, (sc, s, d, family_key, meta) in enumerate(scored_docs):
            if family_key in seen_families:
                continue
            seen_families.add(family_key)
            selected_indexes.add(selected_index)
            formatted_results.append(f"--- ALLIKAS: {s} ---\n{d}")
            if len(formatted_results) >= limit:
                break
            
        context = "\n\n".join(formatted_results)
        if return_debug:
            return context, {
                "fetch_k": fetch_k,
                "candidates": [
                    {
                        "rank": i + 1,
                        "score": round(sc, 4),
                        "source": s,
                        "selected": i in selected_indexes,
                        "metadata": m,
                        "text": d,
                    }
                    for i, (sc, s, d, family_key, m) in enumerate(scored_docs)
                ],
            }
        return context
        
    except Exception as e:
        print(f"VIGA konteksti loomisel: {e}")
        if return_debug:
            return "", {"error": str(e), "fetch_k": None, "candidates": []}
        return ""

def ask_ollama(
    model,
    prompt,
    threads,
    timeout,
    num_predict=1024,
    response_format=None,
    stop=None,
):
    """Saadab päringu Ollama API-le genereerimiseks."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_thread": int(threads),
            "num_predict": int(num_predict),
            "temperature": 0  # Deterministlik vastus on juriidikas kriitiline
        }
    }
    if response_format:
        payload["format"] = response_format
    if stop:
        payload["options"]["stop"] = stop
    
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=timeout)
        if response.status_code == 200:
            return response.json().get("response", "")
        return f"VIGA: Ollama vastas koodiga {response.status_code}"
    except requests.exceptions.Timeout:
        return f"VIGA: Aegumine ({timeout}s)."
    except Exception as e:
        return f"VIGA: Sidekatkestus - {str(e)}"


def ask_gemini(model, prompt, timeout, max_output_tokens=128, api_key=None, response_schema=None):
    """
    Saadab päringu Google Gemini API-le.
    Eeldab, et GEMINI_API_KEY on keskkonnamuutujas.
    """
    effective_key = (api_key or GEMINI_API_KEY or "").strip()
    if not effective_key:
        return "VIGA: GEMINI_API_KEY puudub."

    model_name = str(model or "").strip() or "gemini-2.5-flash"
    url = f"{GEMINI_BASE_URL}/models/{model_name}:generateContent?key={effective_key}"

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": int(max_output_tokens),
            "responseMimeType": "application/json",
            "thinkingConfig": {
                "thinkingBudget": 0
            },
        },
    }
    if response_schema:
        payload["generationConfig"]["responseSchema"] = response_schema

    try:
        response = requests.post(url, json=payload, timeout=timeout)
        if response.status_code != 200:
            err_msg = ""
            try:
                err_json = response.json()
                err_obj = err_json.get("error", {}) if isinstance(err_json, dict) else {}
                err_msg = str(err_obj.get("message", "")).strip()
            except Exception:
                err_msg = ""
            if err_msg:
                return f"VIGA: Gemini vastas koodiga {response.status_code} - {err_msg}"
            return f"VIGA: Gemini vastas koodiga {response.status_code}"

        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return "VIGA: Gemini ei tagastanud kandidaate."

        parts = (
            candidates[0]
            .get("content", {})
            .get("parts", [])
        )
        if not parts:
            return "VIGA: Gemini vastus oli tühi."

        text = parts[0].get("text", "")
        return text or "VIGA: Gemini vastus oli tühi."
    except requests.exceptions.Timeout:
        return f"VIGA: Gemini aegumine ({timeout}s)."
    except Exception as e:
        return f"VIGA: Gemini sidekatkestus - {str(e)}"

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
