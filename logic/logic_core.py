import os
import requests
import json
import re
import unicodedata
from typing import Optional
from datetime import datetime
from zoneinfo import ZoneInfo
import chromadb
from chromadb.utils import embedding_functions
try:
    import oracledb
except Exception:
    oracledb = None

# --- KONFIGURATSIOON ---
# Võtame URL-id keskkonnamuutujatest (kooskõlas docker-compose'iga)
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_API_URL = f"{OLLAMA_URL}/api/generate"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
# Täpne path vastavalt docker-compose volume'ile
CHROMA_DB_PATH = "/app/storage/vector_db"
PROMPTS_FILE = "/app/prompts.json"
ORACLE_ENABLED = str(os.getenv("ORACLE_ENABLED", "no")).strip().lower() in {"1", "true", "yes", "on"}
ORACLE_DSN = os.getenv("ORACLE_DSN", "").strip()
ORACLE_USER = os.getenv("ORACLE_USER", "").strip()
ORACLE_PASSWORD = os.getenv("ORACLE_PASSWORD", "").strip()
ORACLE_CONFIG_DIR = os.path.expanduser(os.getenv("ORACLE_CONFIG_DIR", "").strip())
ORACLE_WALLET_LOCATION = os.path.expanduser(os.getenv("ORACLE_WALLET_LOCATION", "").strip())
ORACLE_WALLET_PASSWORD = os.getenv("ORACLE_WALLET_PASSWORD", "").strip()

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


def resolve_db_backend(requested_backend: Optional[str] = None, default_backend: str = "sqlite"):
    value = str(requested_backend or "").strip().lower()
    if value in {"sqlite", "oracle"}:
        return value
    if value == "auto":
        return "oracle" if ORACLE_ENABLED else "sqlite"
    return default_backend


def get_backend_display_name(db_backend: str):
    return "Oracle" if str(db_backend).lower() == "oracle" else "sqlite/chroma"


def _oracle_connect():
    if oracledb is None:
        raise RuntimeError("Oracle backend requested but python-oracledb is not installed.")
    if not ORACLE_DSN or not ORACLE_USER or not ORACLE_PASSWORD:
        raise RuntimeError("Oracle backend requested but ORACLE_DSN/ORACLE_USER/ORACLE_PASSWORD are missing.")
    kwargs = {"user": ORACLE_USER, "password": ORACLE_PASSWORD, "dsn": ORACLE_DSN}
    if ORACLE_CONFIG_DIR:
        kwargs["config_dir"] = ORACLE_CONFIG_DIR
    if ORACLE_WALLET_LOCATION:
        kwargs["wallet_location"] = ORACLE_WALLET_LOCATION
    if ORACLE_WALLET_PASSWORD:
        kwargs["wallet_password"] = ORACLE_WALLET_PASSWORD
    return oracledb.connect(**kwargs)

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

def _flatten_result_lists(results, key):
    """Tagastab Chroma query tulemused ühe listina ka mitme query_text korral."""
    if not isinstance(results, dict):
        return []
    value = results.get(key)
    if not value:
        return []
    if not isinstance(value, list):
        return []
    if value and isinstance(value[0], list):
        return [item for group in value for item in group]
    return value

def is_secret_metadata(meta):
    """Demo access flag: only explicit classification_level=secret is restricted."""
    return str((meta or {}).get("classification_level", "")).strip().lower() == "secret"

def is_estonian_personal_code(value):
    value = str(value or "").strip()
    # Kursusetöö testandmetes kasutatakse ka fiktiivseid isikukoode.
    # Seetõttu maskeerime kõik Eesti isikukoodi kujuga 11-kohalised väärtused,
    # mitte ainult kontrollsummaga kehtivad isikukoodid.
    return bool(re.fullmatch(r"[1-8]\d{10}", value))

def mask_personal_codes_in_text(text, replacement="***"):
    text = str(text or "")

    def replace_labeled_identifier(match):
        return f"{match.group(1)}{replacement}"

    text = re.sub(
        r"(\b(?:isikukood|isiku\s*kood|personal\s*code|personal_code|national\s*id|national_id)\b\s*:?\s*)[^\s,;.)\]}]+",
        replace_labeled_identifier,
        text,
        flags=re.IGNORECASE,
    )

    def replace_match(match):
        value = match.group(0)
        return replacement if is_estonian_personal_code(value) else value

    return re.sub(r"\b[1-8]\d{10}\b", replace_match, text)

def mask_personal_codes(value, replacement="***"):
    if isinstance(value, str):
        return mask_personal_codes_in_text(value, replacement)
    if isinstance(value, list):
        return [mask_personal_codes(item, replacement) for item in value]
    if isinstance(value, dict):
        is_private_person = (
            str(value.get("counterparty_type", "")).strip().lower() == "private_person"
            or str(value.get("subject_type", "")).strip().lower() == "private_person"
        )
        return {
            key: (
                replacement
                if is_private_person and key in {"subject_id", "counterparty_id", "personal_id"}
                else mask_personal_codes(item, replacement)
            )
            for key, item in value.items()
        }
    return value

def _normal_id_set(values):
    if not values:
        return set()
    return {str(value).strip() for value in values if str(value).strip()}

def is_contract_metadata(meta):
    meta = meta or {}
    return (
        str(meta.get("doc_type", "")).strip().lower() == "contract"
        or str(meta.get("type", "")).strip().lower() == "leping"
        or str(meta.get("chunk_type", "")).strip().lower() == "contract_section"
    )

def _canonical_text(value):
    text = str(value or "").lower()
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))

def _text_words(value):
    return {
        word
        for word in re.findall(r"\w+", _canonical_text(value))
        if len(word) > 1
    }

def _mentioned_contract_ids(query, metas):
    query_l = _canonical_text(query)
    query_words_set = _text_words(query_l)
    mentioned = set()
    for meta in metas:
        if not is_contract_metadata(meta):
            continue
        contract_id = str(meta.get("contract_id", "")).strip()
        if contract_id and _canonical_text(contract_id) in query_l:
            mentioned.add(contract_id)
            continue

        subject_id = str(meta.get("subject_id", "")).strip()
        counterparty_id = str(meta.get("counterparty_id", "")).strip()
        if contract_id and any(identifier and identifier in query_l for identifier in (subject_id, counterparty_id)):
            mentioned.add(contract_id)
            continue

        subject_name = str(meta.get("subject_name", "")).strip()
        subject_words = {word for word in _text_words(subject_name) if len(word) > 2}
        overlap = subject_words & query_words_set
        if (
            contract_id
            and subject_words
            and (
                subject_words.issubset(query_words_set)
                or len(overlap) >= min(2, len(subject_words))
            )
        ):
            mentioned.add(contract_id)
    return mentioned

def _contract_catalog_metas():
    """Loeb lepingute metadata, et nimepõhine tuvastus ei sõltuks vektorotsingu esimestest tabamustest."""
    try:
        results = collection.get(
            where={"doc_type": "contract"},
            include=["metadatas"],
        )
    except Exception:
        results = {}

    metas = results.get("metadatas") or []
    if isinstance(metas, list) and metas:
        return [meta for meta in metas if isinstance(meta, dict) and is_contract_metadata(meta)]

    try:
        results = collection.get(include=["metadatas"], limit=5000)
    except TypeError:
        results = collection.get(include=["metadatas"])
    except Exception:
        return []

    metas = results.get("metadatas") or []
    if not isinstance(metas, list):
        return []
    return [meta for meta in metas if isinstance(meta, dict) and is_contract_metadata(meta)]

def _contract_section_intent_boost(meta, query_words):
    section_title = str((meta or {}).get("section_title", "")).lower()
    section_title_canonical = _canonical_text(section_title)
    section_words = _text_words(section_title)
    query_word_set = set(query_words)
    query_stems = {word[:4] for word in query_word_set if len(word) >= 4}
    section_stems = {word[:4] for word in section_words if len(word) >= 4}
    boost = 0.35 * len(section_words & query_word_set)
    boost += 0.45 * len(section_stems & query_stems)
    if "tasu" in query_word_set and "tasu" in section_title:
        boost += 1.4
    if "sisu" in query_word_set and "sisu" in section_title:
        boost += 1.6
    if any(word.startswith("sisu") for word in query_word_set) and "sisu" in section_title:
        boost += 1.0
    if ("too" in query_word_set or "toode" in query_word_set) and "too" in section_title_canonical:
        boost += 1.0
    if "töö" in query_word_set and "töö" in section_title:
        boost += 1.0
    if "lepingupool" in section_title or "lepingupooled" in section_title:
        boost += 0.2
    if (
        any(word.startswith("sisu") or word.startswith("töö") for word in query_word_set)
        and ("allkirjad" in section_title or "lepingupooled" in section_title)
    ):
        boost -= 0.8
    return boost

def _append_contract_siblings(scored_docs, contract_ids, seen_snippets, query_words):
    if not contract_ids:
        return scored_docs

    existing_keys = {
        (
            str((meta or {}).get("contract_id", "")),
            str((meta or {}).get("section_index", "")),
        )
        for *_rest, meta in scored_docs
        if is_contract_metadata(meta)
    }

    for contract_id in contract_ids:
        try:
            results = collection.get(
                where={"contract_id": contract_id},
                include=["documents", "metadatas"],
            )
        except Exception:
            continue

        docs = results.get("documents") or []
        metas = results.get("metadatas") or []
        for doc, meta in zip(docs, metas):
            if not isinstance(doc, str) or not doc.strip() or not isinstance(meta, dict):
                continue
            key = (str(meta.get("contract_id", "")), str(meta.get("section_index", "")))
            if key in existing_keys:
                continue
            snippet = doc[:100].strip().lower()
            if snippet in seen_snippets:
                continue
            existing_keys.add(key)
            seen_snippets.add(snippet)
            source = meta.get("source") or meta.get("contract_id") or "LEPING"
            family_key = ("contract", str(meta.get("contract_id", "")), str(meta.get("section_index", "")))
            section_words = _text_words(meta.get("section_title", ""))
            shared_section_words = section_words & set(query_words)
            sibling_score = 1.3 + 0.35 * len(shared_section_words)
            section_title = str(meta.get("section_title", "")).lower()
            if "tasu" in query_words and "tasu" in section_title:
                sibling_score += 1.2
            if "sisu" in query_words and "sisu" in section_title:
                sibling_score += 1.0
            if "töö" in query_words and "töö" in section_title:
                sibling_score += 0.6
            if "pool" in section_title or "lepingupool" in section_title:
                sibling_score += 0.2
            scored_docs.append((sibling_score, source, doc, family_key, meta))
    return scored_docs

def _merge_query_results(base_results, extra_results):
    """Liidab kaks Chroma query tulemust üheks key->[list-of-lists] struktuuriks."""
    merged = {}
    for key in ("documents", "metadatas", "distances", "ids"):
        base_value = (base_results or {}).get(key) or []
        extra_value = (extra_results or {}).get(key) or []
        if isinstance(base_value, list) and base_value and isinstance(base_value[0], list):
            base_groups = base_value
        elif isinstance(base_value, list):
            base_groups = [base_value]
        else:
            base_groups = [[]]
        if isinstance(extra_value, list) and extra_value and isinstance(extra_value[0], list):
            extra_groups = extra_value
        elif isinstance(extra_value, list):
            extra_groups = [extra_value]
        else:
            extra_groups = [[]]

        max_len = max(len(base_groups), len(extra_groups))
        groups = []
        for idx in range(max_len):
            left = base_groups[idx] if idx < len(base_groups) and isinstance(base_groups[idx], list) else []
            right = extra_groups[idx] if idx < len(extra_groups) and isinstance(extra_groups[idx], list) else []
            groups.append(left + right)
        merged[key] = groups
    return merged

def get_candidate_filter_reason(
    meta,
    secret_allowed=False,
    allowed_subject_ids=None,
    allowed_tenant_ids=None,
    allow_all_subjects=False,
):
    meta = meta or {}
    if is_secret_metadata(meta) and not secret_allowed:
        return "secret_not_allowed"

    allowed_tenants = _normal_id_set(allowed_tenant_ids)
    tenant_id = str(meta.get("tenant_id", "")).strip()
    if allowed_tenants and tenant_id and tenant_id not in allowed_tenants:
        return "tenant_not_allowed"

    if is_contract_metadata(meta) and not allow_all_subjects:
        allowed_subjects = _normal_id_set(allowed_subject_ids)
        subject_id = str(meta.get("subject_id", "")).strip()
        if subject_id and subject_id not in allowed_subjects:
            return "subject_not_allowed"

    return ""

def format_debug_candidates(
    scored_docs,
    selected_indexes,
    secret_allowed=False,
    allowed_subject_ids=None,
    allowed_tenant_ids=None,
    allow_all_subjects=False,
    allow_personal_data=False,
):
    candidates = []
    for i, (sc, s, d, _family_key, m) in enumerate(scored_docs):
        is_secret = is_secret_metadata(m)
        filter_reason = get_candidate_filter_reason(
            m,
            secret_allowed=secret_allowed,
            allowed_subject_ids=allowed_subject_ids,
            allowed_tenant_ids=allowed_tenant_ids,
            allow_all_subjects=allow_all_subjects,
        )
        visible_meta = m if allow_personal_data else mask_personal_codes(m)
        visible_text = d if allow_personal_data else mask_personal_codes_in_text(d)
        item = {
            "rank": i + 1,
            "score": round(sc, 4),
            "source": s,
            "selected": i in selected_indexes,
            "metadata": visible_meta,
            "text": visible_text,
            "is_secret": is_secret,
            "filtered": False,
        }
        if filter_reason:
            item["filtered"] = True
            item["filtered_reason"] = filter_reason
        candidates.append(item)
    return candidates


def get_context_oracle(
    query,
    n_results=5,
    max_context_blocks=3,
    return_debug=False,
    secret=False,
    allowed_subject_ids=None,
    allowed_tenant_ids=None,
    allow_all_subjects=False,
    allow_personal_data=False,
    original_query=None,
):
    try:
        query_text = str(query or "").strip()
        if not query_text:
            if return_debug:
                return "", {"fetch_k": 0, "candidates": []}
            return ""

        query_vec = embedding_func([query_text])[0]
        vec_json = json.dumps(query_vec.tolist() if hasattr(query_vec, "tolist") else query_vec, separators=(",", ":"))
        fetch_k = max(int(n_results or 5), int(max_context_blocks or 3), 5) * 4

        sql = """
            SELECT source_key, content, metadata_json, dist
            FROM (
                SELECT
                    s.source_key AS source_key,
                    c.content AS content,
                    c.metadata_json AS metadata_json,
                    VECTOR_DISTANCE(e.embedding, TO_VECTOR(:vec_json), COSINE) AS dist
                FROM rag_embeddings e
                JOIN rag_chunks c ON c.chunk_id = e.chunk_id
                JOIN rag_sources s ON s.source_id = c.source_id
                WHERE e.embedding_model = :embedding_model
                ORDER BY dist
            )
            WHERE ROWNUM <= :fetch_k
        """

        rows = []
        with _oracle_connect() as conn:
            cur = conn.cursor()
            cur.execute(
                sql,
                {
                    "vec_json": vec_json,
                    "embedding_model": EMBEDDING_MODEL,
                    "fetch_k": int(fetch_k),
                },
            )
            raw_rows = cur.fetchall() or []
            for row in raw_rows:
                source_key = str(row[0] or "RAG")
                raw_doc = row[1]
                raw_meta = row[2]
                dist = float(row[3]) if row[3] is not None else 1.4

                doc_text = raw_doc.read() if hasattr(raw_doc, "read") else str(raw_doc or "")
                meta_text = raw_meta.read() if hasattr(raw_meta, "read") else str(raw_meta or "{}")
                rows.append((source_key, doc_text, meta_text, dist))

        scoring_query = query_text.lower()
        query_words = list(_text_words(scoring_query))
        query_numbers = re.findall(r'\d[\d\s]*', scoring_query)
        query_stems = {word[:5] for word in query_words if len(word) >= 6}
        contract_intent = any(
            token in scoring_query
            for token in ["leping", "lepingu", "lepingud", "tasu", "töö sisu", "too sisu", "toode sisu", "subject_id"]
        )

        scored_docs = []
        for idx, row in enumerate(rows):
            source_key = row[0]
            doc = row[1] or ""
            # Normaliseeri levinud mojibake variandid, et §-põhised kontrollid töötaksid.
            doc = (
                doc.replace("Ā§", "§")
                .replace("Â§", "§")
                .replace("ÃÂ§", "§")
            )
            meta_raw = row[2]
            dist = row[3]
            try:
                raw_meta_text = meta_raw or "{}"
                if not isinstance(raw_meta_text, str):
                    raw_meta_text = str(raw_meta_text)
                meta = json.loads(raw_meta_text)
            except Exception:
                meta = {}
            if not isinstance(meta, dict):
                meta = {}
            v_score = max(0, 1.4 - dist)
            k_score = 0.0
            doc_lower = doc.lower()
            display_name = str(meta.get("display_name", "")).lower()
            para_title = str(meta.get("paragraph_title", "")).lower()
            section_title = str(meta.get("section_title", "")).lower()
            chunk_type = str(meta.get("chunk_type", "")).lower()

            title_text = f"{display_name} {para_title} {section_title}".strip()
            title_words = set(re.findall(r'\w+', title_text))
            title_stems = {word[:5] for word in title_words if len(word) >= 6}
            doc_words = set(re.findall(r'\w+', doc_lower))
            doc_stems = {word[:5] for word in doc_words if len(word) >= 6}

            for word in query_words:
                if len(word) > 4 and word in doc_lower:
                    k_score += 0.4
                if len(word) > 4 and word in para_title:
                    k_score += 0.45
                if len(word) > 4 and word in section_title:
                    k_score += 0.4
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

            if chunk_type == "subsection":
                k_score += 0.3
            elif chunk_type == "section":
                k_score += 0.15
            elif chunk_type == "point":
                k_score -= 0.05

            score = v_score + k_score
            if contract_intent and is_contract_metadata(meta):
                score += 0.65 + _contract_section_intent_boost(meta, query_words)

            family_key = (
                str(meta.get("law", "")),
                str(meta.get("section", "")),
                str(meta.get("subsection", "")),
                str(meta.get("section_index", "")),
                str(idx),
            )
            scored_docs.append((score, source_key, doc, family_key, meta))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        secret_allowed = bool(secret)
        limit = max(1, int(max_context_blocks or 3))

        formatted_results = []
        selected_indexes = set()
        seen_families = set()
        for selected_index, (sc, source, doc, family_key, meta) in enumerate(scored_docs):
            if get_candidate_filter_reason(
                meta,
                secret_allowed=secret_allowed,
                allowed_subject_ids=allowed_subject_ids,
                allowed_tenant_ids=allowed_tenant_ids,
                allow_all_subjects=allow_all_subjects,
            ):
                continue
            if family_key in seen_families:
                continue
            seen_families.add(family_key)
            selected_indexes.add(selected_index)
            visible_doc = doc if allow_personal_data else mask_personal_codes_in_text(doc)
            formatted_results.append(f"--- ALLIKAS: {source} ---\n{visible_doc}")
            if len(formatted_results) >= limit:
                break

        context = "\n\n".join(formatted_results)
        if return_debug:
            return context, {
                "fetch_k": fetch_k,
                "query_texts": [query_text] + ([str(original_query).strip()] if str(original_query or "").strip() else []),
                "target_contract_ids": [],
                "contract_intent": bool(contract_intent),
                "contract_probe_added": False,
                "secret": secret_allowed,
                "allowed_subject_ids": list(_normal_id_set(allowed_subject_ids)),
                "allowed_tenant_ids": list(_normal_id_set(allowed_tenant_ids)),
                "allow_all_subjects": bool(allow_all_subjects),
                "allow_personal_data": bool(allow_personal_data),
                "candidates": format_debug_candidates(
                    scored_docs,
                    selected_indexes,
                    secret_allowed,
                    allowed_subject_ids=allowed_subject_ids,
                    allowed_tenant_ids=allowed_tenant_ids,
                    allow_all_subjects=allow_all_subjects,
                    allow_personal_data=allow_personal_data,
                ),
            }
        return context
    except Exception as e:
        print(f"VIGA Oracle konteksti loomisel: {e}")
        if return_debug:
            return "", {"error": str(e), "fetch_k": None, "candidates": []}
        return ""

def get_context(
    query,
    n_results=5,
    max_context_blocks=3,
    return_debug=False,
    secret=False,
    allowed_subject_ids=None,
    allowed_tenant_ids=None,
    allow_all_subjects=False,
    allow_personal_data=False,
    original_query=None,
    db_backend="sqlite",
):
    """
    Teostab RAG-otsingu koos hübriidse skoorimisega (Vektor + Märksõnad).
    Optimeeritud bge-m3 distantsidele.
    """
    try:
        selected_backend = resolve_db_backend(db_backend, default_backend="sqlite")
        if selected_backend == "oracle":
            return get_context_oracle(
                query,
                n_results=n_results,
                max_context_blocks=max_context_blocks,
                return_debug=return_debug,
                secret=secret,
                allowed_subject_ids=allowed_subject_ids,
                allowed_tenant_ids=allowed_tenant_ids,
                allow_all_subjects=allow_all_subjects,
                allow_personal_data=allow_personal_data,
                original_query=original_query,
            )
        # Küsime vektorbaasist rohkem kandidaate kui lõppväljundisse vaja,
        # et Pythonis tehtav ümberreastamine ja dedupe saaks päriselt mõjuda.
        fetch_k = max(int(n_results or 5), int(max_context_blocks or 3), 5) * 4
        query_texts = [str(query or "").strip()]
        original_query_text = str(original_query or "").strip()
        if original_query_text and original_query_text not in query_texts:
            query_texts.append(original_query_text)
        query_texts = [text for text in query_texts if text]
        if not query_texts:
            if return_debug:
                return "", {"fetch_k": fetch_k, "candidates": []}
            return ""

        results = collection.query(query_texts=query_texts, n_results=fetch_k)

        contract_intent = any(
            token in " ".join(query_texts).lower()
            for token in ["leping", "lepingu", "lepingud", "tasu", "töö sisu", "toĢoĢ sisu", "toode sisu", "subject_id"]
        )
        contract_probe_added = False
        if contract_intent:
            contract_results = None
            for contract_where in ({"doc_type": "contract"}, {"type": "leping"}, {"chunk_type": "contract_section"}):
                try:
                    contract_results = collection.query(
                        query_texts=query_texts,
                        n_results=max(fetch_k, 30),
                        where=contract_where,
                    )
                except Exception:
                    contract_results = None
                if _flatten_result_lists(contract_results or {}, "documents"):
                    break
            if _flatten_result_lists(contract_results or {}, "documents"):
                results = _merge_query_results(results, contract_results)
                contract_probe_added = True
        
        docs = _flatten_result_lists(results, "documents")
        if not docs:
            if return_debug:
                return "", {"fetch_k": fetch_k, "candidates": []}
            return ""

        metas = _flatten_result_lists(results, "metadatas")
        distances = _flatten_result_lists(results, "distances")
        if len(metas) < len(docs):
            metas = metas + [{} for _ in range(len(docs) - len(metas))]
        if len(distances) < len(docs):
            distances = distances + [1.4 for _ in range(len(docs) - len(distances))]
        
        scoring_query = " ".join(query_texts).lower()
        query_words = list(_text_words(scoring_query))
        query_numbers = re.findall(r'\d[\d\s]*', scoring_query)
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

            for exact_query in {text.lower() for text in query_texts if text.strip()}:
                if exact_query in doc_lower:
                    k_score += 0.5
                if exact_query in para_title:
                    k_score += 0.4
                if exact_query in display_name:
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
            if is_contract_metadata(meta):
                family_key = (
                    "contract",
                    str(meta.get("contract_id", "")),
                    str(meta.get("section_index", "")),
                )
            else:
                family_key = (
                    str(meta.get("law", "")),
                    str(meta.get("section", "")),
                    str(meta.get("subsection", "")),
                )
            scored_docs.append((final_score, source, doc, family_key, meta))

        # Sorteerime tulemused lõpliku hübriidse skoori järgi
        entity_query = original_query or query
        contract_catalog_metas = _contract_catalog_metas()
        target_contract_ids = (
            _mentioned_contract_ids(entity_query, metas)
            | _mentioned_contract_ids(entity_query, contract_catalog_metas)
        )
        scored_docs = _append_contract_siblings(
            scored_docs,
            target_contract_ids,
            seen_snippets,
            query_words,
        )
        if target_contract_ids:
            boosted_docs = []
            for sc, source, doc, family_key, meta in scored_docs:
                contract_id = str((meta or {}).get("contract_id", ""))
                if is_contract_metadata(meta) and contract_id in target_contract_ids:
                    sc += 0.8 + _contract_section_intent_boost(meta, query_words)
                boosted_docs.append((sc, source, doc, family_key, meta))
            scored_docs = boosted_docs
        elif contract_intent:
            boosted_docs = []
            for sc, source, doc, family_key, meta in scored_docs:
                if is_contract_metadata(meta):
                    sc += 0.65 + _contract_section_intent_boost(meta, query_words)
                boosted_docs.append((sc, source, doc, family_key, meta))
            scored_docs = boosted_docs

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        secret_allowed = bool(secret)
        selectable_docs = [
            item for item in scored_docs
            if not get_candidate_filter_reason(
                item[4],
                secret_allowed=secret_allowed,
                allowed_subject_ids=allowed_subject_ids,
                allowed_tenant_ids=allowed_tenant_ids,
                allow_all_subjects=allow_all_subjects,
            )
            and (
                not target_contract_ids
                or not is_contract_metadata(item[4])
                or str((item[4] or {}).get("contract_id", "")) in target_contract_ids
            )
            and (
                not target_contract_ids
                or is_contract_metadata(item[4])
            )
        ]

        # FAIL FAST lävend: bge-m3 puhul on 0.65-0.7 turvaline piir
        # See hoiab ära hallutsinatsioonid tühja konteksti baasilt
        if not selectable_docs or selectable_docs[0][0] < 0.65:
            if return_debug:
                return "", {
                    "fetch_k": fetch_k,
                    "query_texts": query_texts,
                    "target_contract_ids": sorted(target_contract_ids),
                    "contract_intent": bool(contract_intent),
                    "contract_probe_added": bool(contract_probe_added),
                    "secret": secret_allowed,
                    "allowed_subject_ids": list(_normal_id_set(allowed_subject_ids)),
                    "allowed_tenant_ids": list(_normal_id_set(allowed_tenant_ids)),
                    "allow_all_subjects": bool(allow_all_subjects),
                    "allow_personal_data": bool(allow_personal_data),
                    "candidates": format_debug_candidates(
                        scored_docs,
                        set(),
                        secret_allowed,
                        allowed_subject_ids=allowed_subject_ids,
                        allowed_tenant_ids=allowed_tenant_ids,
                        allow_all_subjects=allow_all_subjects,
                        allow_personal_data=allow_personal_data,
                    ),
                }
            return ""

        formatted_results = []
        seen_families = set()
        selected_indexes = set()
        limit = max(1, int(max_context_blocks or 3))
        for selected_index, (sc, s, d, family_key, meta) in enumerate(scored_docs):
            if (
                target_contract_ids
                and is_contract_metadata(meta)
                and str((meta or {}).get("contract_id", "")) not in target_contract_ids
            ):
                continue
            if target_contract_ids and not is_contract_metadata(meta):
                continue
            if get_candidate_filter_reason(
                meta,
                secret_allowed=secret_allowed,
                allowed_subject_ids=allowed_subject_ids,
                allowed_tenant_ids=allowed_tenant_ids,
                allow_all_subjects=allow_all_subjects,
            ):
                continue
            if family_key in seen_families:
                continue
            seen_families.add(family_key)
            selected_indexes.add(selected_index)
            visible_doc = d if allow_personal_data else mask_personal_codes_in_text(d)
            formatted_results.append(f"--- ALLIKAS: {s} ---\n{visible_doc}")
            if len(formatted_results) >= limit:
                break
            
        context = "\n\n".join(formatted_results)
        if return_debug:
            return context, {
                "fetch_k": fetch_k,
                "query_texts": query_texts,
                "target_contract_ids": sorted(target_contract_ids),
                "contract_intent": bool(contract_intent),
                "contract_probe_added": bool(contract_probe_added),
                "secret": secret_allowed,
                "allowed_subject_ids": list(_normal_id_set(allowed_subject_ids)),
                "allowed_tenant_ids": list(_normal_id_set(allowed_tenant_ids)),
                "allow_all_subjects": bool(allow_all_subjects),
                "allow_personal_data": bool(allow_personal_data),
                "candidates": format_debug_candidates(
                    scored_docs,
                    selected_indexes,
                    secret_allowed,
                    allowed_subject_ids=allowed_subject_ids,
                    allowed_tenant_ids=allowed_tenant_ids,
                    allow_all_subjects=allow_all_subjects,
                    allow_personal_data=allow_personal_data,
                ),
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

