from fastapi import FastAPI, Depends, HTTPException, status, Query
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import logic_core
import json
import os
import time
import re
from datetime import datetime
from zoneinfo import ZoneInfo
from pydantic import BaseModel, Field
from typing import Optional, List, Literal

# --- KONFIGURATSIOON ---
app = FastAPI(
    title="Hankija AI Turvakiht API", 
    version="1.9", 
    description="REST liides tehisintellekti turvakihi ja valideerimise kasutamiseks koos kohandatavate parameetritega",
    docs_url="/swagger"
)
security = HTTPBasic()

# Failide asukohad Dockeris
API_LOG_FILE = "/app/api_access.log"
UI_LOG_FILE = "/app/ai_turvakiht.log"
TEST_LOG_FILES = {
    "test-pre-check": "/testing/bench-pre-check-log.json",
    "test-post-check": "/testing/bench-post-check-log.json",
    "test-post-check-use-cases": "/testing/post-check-use-cases-log.json",
    "test-llm": "/testing/llm-test-log.json",
    "test-retrieval": "/testing/retr-test-log.json",
    "test-stability": "/testing/stability-test-log.json",
    "test-benchmark-embeddings": "/testing/benchmark_embeddings-log.json",
    "test-normalizer": "/testing/normalizer-test-log.json",
    "prompts-change": "/app/prompts_change_log.json",
}


def ee_now_str(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Tagastab kellaaja Eesti ajavööndis."""
    return datetime.now(ZoneInfo("Europe/Tallinn")).strftime(fmt)

# --- AUTENTIMINE ---
def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    """Põhiline HTTP Basic autentimine."""
    if credentials.username != "admin" or credentials.password != "parool":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Vale kasutajanimi või parool",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# --- ANDMEMUDELID ---
class PreCheckRequest(BaseModel):
    user_input: str
    model: str = "gemma2:2b"
    normalization_mode: Literal["precheck", "external", "off"] = "precheck"
    timeout: Optional[int] = 60
    threads: Optional[int] = 4

class NormalizeRequest(BaseModel):
    user_input: str
    model: str = "alarjoeste/estonian-normalizer"
    gemini_api_key: Optional[str] = None
    timeout: Optional[int] = 90
    threads: Optional[int] = 4

class MainQueryRequest(BaseModel):
    user_input: str
    context: str
    model: str = "llama3:8b"
    timeout: Optional[int] = 120
    threads: Optional[int] = 4

class RetrievalRequest(BaseModel):
    query: str = Field(..., min_length=1)
    original_query: Optional[str] = None
    n_results: Optional[int] = Field(9, ge=1, le=25)
    max_context_blocks: Optional[int] = Field(5, ge=1, le=25)
    secret: Optional[bool] = False
    allowed_subject_ids: List[str] = Field(default_factory=list)
    allowed_tenant_ids: List[str] = Field(default_factory=list)
    allow_all_subjects: Optional[bool] = False
    allow_personal_data: Optional[bool] = False

class PostCheckRequest(BaseModel):
    ai_response: str
    # Pre-check "user_input" (algne, töötlemata küsimus)
    original_user_input: Optional[str] = Field(None, min_length=1)
    # Pre-check väljund (normaliseeritud küsimus), mille alusel tehti põhiküsimus/RAG
    normalized_query: Optional[str] = ""
    # Kontekst, mille alusel põhipäring LLM-s tehti (RAG kontekst)
    context: Optional[str] = ""
    model: str = "gemma2:2b"
    quality_model: Optional[str] = None
    security_model: Optional[str] = None
    timeout: Optional[int] = 90
    threads: Optional[int] = 4
    secret: Optional[bool] = False
    allowed_subject_ids: List[str] = Field(default_factory=list)
    allowed_tenant_ids: List[str] = Field(default_factory=list)
    allow_all_subjects: Optional[bool] = False
    allow_personal_data: Optional[bool] = False
    sources_returned_raw: List[dict] = Field(default_factory=list)

# --- LOGIMINE ---
def log_api_call(endpoint: str, status_code: int, duration: float, user: str, extra_data: dict = None):
    """Logib API päringu info faili."""
    log_entry = {
        "timestamp": ee_now_str(),
        "endpoint": endpoint,
        "user": user,
        "status": status_code,
        "duration_sec": round(duration, 3)
    }
    if extra_data:
        log_entry.update(extra_data)
    log_entry = logic_core.mask_personal_codes(log_entry)
        
    try:
        with open(API_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        print(f"Logimisviga: {e}")

# --- API ENDPOINTID ---

@app.post("/pre-check", tags=["Valideerimine"])
async def pre_check(req: PreCheckRequest, user: str = Depends(authenticate)):
    """
    Kontrollib sisendi turvalisust ja sobivust enne põhipäringut.
    Võimaldab määrata mudelit, timeouti ja lõimede arvu.
    """
    start_time = time.time()
    start_time_str = time.strftime("%H:%M:%S")
    try:
        # Võtame õige võtmega prompti failist
        use_precheck_normalization = req.normalization_mode == "precheck"
        prompt_key = "PRE_CHECK_PROMPT" if use_precheck_normalization else "PRE_CHECK_SECURITY_ONLY_PROMPT"
        sys_prompt = logic_core.PROMPTS.get(prompt_key)
        if not sys_prompt:
            raise RuntimeError(f"Prompt puudub: {prompt_key}")
        # Asendame promptis oleva muutuja tegeliku sisendiga
        full_prompt = sys_prompt.replace("{u_input}", req.user_input)
        
        result = logic_core.ask_ollama(
            req.model,
            full_prompt,
            req.threads,
            req.timeout
        )
        duration = time.time() - start_time
        
        # Parsime Ollama string vastuse reaalseks JSONiks
        parsed_result = logic_core.parse_json_res(result)
        
        status_value = parsed_result.get("status", "BLOCKED")
        normalized_value = parsed_result.get("normalized_query", "") if use_precheck_normalization else req.user_input
        reason_value = parsed_result.get("reason", "")
        if not use_precheck_normalization and status_value != "ALLOWED" and not reason_value:
            reason_value = "Päring ei läbinud turvakontrolli."

        # Kui kombineeritud pre-check (turva + normaliseerimine) annab BLOCKED
        # ilma sisulise põhjenduseta, teeme teise hinnangu security-only promptiga.
        # See vähendab valepositiivseid blokeeringuid lihtsate domeeniküsimuste puhul.
        if use_precheck_normalization and status_value == "BLOCKED" and not reason_value:
            sec_prompt = logic_core.PROMPTS.get("PRE_CHECK_SECURITY_ONLY_PROMPT")
            if sec_prompt:
                sec_full_prompt = sec_prompt.replace("{u_input}", req.user_input)
                sec_result = logic_core.ask_ollama(
                    req.model,
                    sec_full_prompt,
                    req.threads,
                    req.timeout
                )
                sec_parsed = logic_core.parse_json_res(sec_result)
                sec_status = sec_parsed.get("status", "BLOCKED")
                sec_reason = sec_parsed.get("reason", "")
                if sec_status == "ALLOWED":
                    status_value = "ALLOWED"
                    reason_value = ""
                elif sec_reason:
                    reason_value = sec_reason

        response_data = {
            "model": req.model,
            "normalization_mode": req.normalization_mode,
            "prompt_key": prompt_key,
            "prompt": full_prompt,
            "start_time": start_time_str,
            "status": status_value,
            "normalized": normalized_value,
            "reason": reason_value,
            "normalization_applied": use_precheck_normalization,
            "duration": round(duration * 1000, 2),
            "raw_response": result
        }
        
        log_api_call("/pre-check", 200, duration, user, response_data)
        return response_data
    except Exception as e:
        log_api_call("/pre-check", 500, 0, user, {"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/normalize", tags=["Valideerimine"])
async def normalize_query(req: NormalizeRequest, user: str = Depends(authenticate)):
    """
    Normaliseerib kasutaja sisendi eraldi normaliseerimismudeli abil.
    """
    start_time = time.time()
    start_time_str = time.strftime("%H:%M:%S")
    try:
        model_name = str(req.model or "").strip()
        is_gemini = model_name.lower().startswith("gemini:")
        prompt_key = "NORMALIZE_QUERY_PROMPT_GEMINI" if is_gemini else "NORMALIZE_QUERY_PROMPT"
        sys_prompt = logic_core.PROMPTS.get(prompt_key)
        if not sys_prompt:
            raise RuntimeError(f"Prompt puudub: {prompt_key}")
        full_prompt = sys_prompt.replace("{u_input}", req.user_input)

        if is_gemini:
            gemini_model = model_name.split(":", 1)[1].strip() or "gemini-2.5-flash"
            result = logic_core.ask_gemini(
                gemini_model,
                full_prompt,
                req.timeout,
                max_output_tokens=128,
                api_key=req.gemini_api_key,
                response_schema={
                    "type": "OBJECT",
                    "properties": {
                        "normalized_query": {"type": "STRING"}
                    },
                    "required": ["normalized_query"]
                },
            )
        else:
            result = logic_core.ask_ollama(
                req.model,
                full_prompt,
                req.threads,
                req.timeout,
                num_predict=96,
                response_format="json",
                stop=["\n\n###", "\n###", "\nUSER INPUT:", "\nJSON OUTPUT:"],
            )
        duration = time.time() - start_time
        if isinstance(result, str) and result.startswith("VIGA:"):
            error_text = str(result)
            status_code = 502
            error_lower = error_text.lower()
            if "aegumine" in error_lower:
                status_code = 504
            elif "puudub" in error_lower:
                status_code = 400
            else:
                code_match = re.search(r"koodiga\s+(\d{3})", error_text)
                if code_match:
                    status_code = int(code_match.group(1))
            log_api_call(
                "/normalize",
                status_code,
                duration,
                user,
                {
                    "model": req.model,
                    "prompt_key": prompt_key,
                    "prompt": full_prompt,
                    "start_time": start_time_str,
                    "error": error_text,
                },
            )
            raise HTTPException(status_code=status_code, detail=error_text)

        parsed_result = logic_core.parse_json_res(result)
        normalized_value = str(parsed_result.get("normalized_query", "")).strip()
        if not normalized_value:
            raise HTTPException(
                status_code=502,
                detail="Normalize mudel ei tagastanud välja 'normalized_query' korrektses JSON-vormis."
            )

        response_data = {
            "model": req.model,
            "prompt_key": prompt_key,
            "prompt": full_prompt,
            "start_time": start_time_str,
            "normalized": normalized_value,
            "duration": round(duration * 1000, 2),
            "raw_response": result
        }

        log_api_call("/normalize", 200, duration, user, response_data)
        return response_data
    except HTTPException:
        raise
    except Exception as e:
        log_api_call("/normalize", 500, 0, user, {"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", tags=["Põhipäring"])
async def run_query(req: MainQueryRequest, user: str = Depends(authenticate)):
    """
    Saadab kasutaja päringu tehisintellektile.
    Konfigureeritav: model, timeout, threads.
    """
    start_time = time.time()
    start_time_str = time.strftime("%H:%M:%S")
    try:
        # Võtame prompti prompts.json failist ning täidame sisendparameetritega
        rag_prompt_template = logic_core.PROMPTS.get("RAG_PROMPT", "{context}\n{query}")
        full_prompt = (
            rag_prompt_template
            .replace("{context}", req.context)
            .replace("{query}", req.user_input)
        )
        
        result = logic_core.ask_ollama(
            req.model,
            full_prompt,
            req.threads,
            req.timeout
        )
        duration = time.time() - start_time
        
        response_data = {
            "model": req.model,
            "prompt": full_prompt,
            "start_time": start_time_str,
            "context_used": req.context,
            "user_input": req.user_input,
            "result": result,
            "duration": round(duration * 1000, 2),
            "raw_response": result
        }
        
        log_api_call("/query", 200, duration, user, response_data)
        return response_data
    except Exception as e:
        log_api_call("/query", 500, 0, user, {"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrieval", tags=["Retrieval"])
async def run_retrieval(req: RetrievalRequest, user: str = Depends(authenticate)):
    """
    Toob vektorbaasist RAG konteksti antud päringu jaoks.
    """
    start_time = time.time()
    start_time_str = time.strftime("%H:%M:%S")
    try:
        n_results = req.n_results or 9
        max_context_blocks = req.max_context_blocks or 5
        secret = bool(req.secret)
        allowed_subject_ids = [str(x).strip() for x in (req.allowed_subject_ids or []) if str(x).strip()]
        allowed_tenant_ids = [str(x).strip() for x in (req.allowed_tenant_ids or []) if str(x).strip()]
        allow_all_subjects = bool(req.allow_all_subjects)
        allow_personal_data = bool(req.allow_personal_data)
        context, retrieval_debug = logic_core.get_context(
            req.query,
            n_results=n_results,
            max_context_blocks=max_context_blocks,
            return_debug=True,
            secret=secret,
            allowed_subject_ids=allowed_subject_ids,
            allowed_tenant_ids=allowed_tenant_ids,
            allow_all_subjects=allow_all_subjects,
            allow_personal_data=allow_personal_data,
            original_query=req.original_query,
        )
        duration = time.time() - start_time
        blocks = [f"--- ALLIKAS:{block}" for block in context.split("--- ALLIKAS:") if block.strip()]

        response_data = {
            "query": req.query,
            "original_query": req.original_query,
            "n_results": n_results,
            "max_context_blocks": max_context_blocks,
            "secret": secret,
            "allowed_subject_ids": allowed_subject_ids,
            "allowed_tenant_ids": allowed_tenant_ids,
            "allow_all_subjects": allow_all_subjects,
            "allow_personal_data": allow_personal_data,
            "start_time": start_time_str,
            "found": bool(context.strip()),
            "context": context,
            "sources_returned": [block[:100] + "..." for block in blocks],
            "sources_returned_raw": retrieval_debug.get("candidates", []),
            "retrieval_debug": {
                "fetch_k": retrieval_debug.get("fetch_k"),
                "query_texts": retrieval_debug.get("query_texts", []),
                "target_contract_ids": retrieval_debug.get("target_contract_ids", []),
                "contract_intent": bool(retrieval_debug.get("contract_intent")),
                "contract_probe_added": bool(retrieval_debug.get("contract_probe_added")),
                "candidate_count": len(retrieval_debug.get("candidates", [])),
                "secret_candidate_count": len([
                    c for c in retrieval_debug.get("candidates", [])
                    if c.get("is_secret")
                ]),
                "filtered_secret_count": len([
                    c for c in retrieval_debug.get("candidates", [])
                    if c.get("filtered_reason") == "secret_not_allowed"
                ]),
                "filtered_subject_count": len([
                    c for c in retrieval_debug.get("candidates", [])
                    if c.get("filtered_reason") == "subject_not_allowed"
                ]),
                "filtered_tenant_count": len([
                    c for c in retrieval_debug.get("candidates", [])
                    if c.get("filtered_reason") == "tenant_not_allowed"
                ]),
                "personal_data_masked": not allow_personal_data,
            },
            "raw_context_preview": (context[:200] + "...") if context else "PUUDUB",
            "duration": round(duration * 1000, 2),
        }

        log_api_call("/retrieval", 200, duration, user, response_data)
        return response_data
    except Exception as e:
        log_api_call("/retrieval", 500, 0, user, {"query": req.query, "error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))


def _run_quality_post_check(req: PostCheckRequest):
    # Hard rule: tühi kontekst + sisuline vastus => BLOCKED
    if not (req.context or "").strip():
        if (req.ai_response or "").strip() != "Esitatud kontekstis info puudub.":
            return {
                "model": "hard-rule",
                "status": "BLOCKED",
                "reason": "Kontekst puudub, kuid vastus sisaldab sisulisi väiteid.",
                "duration": 0.0,
                "prompt": "",
                "raw_response": "",
            }

    # Hard rule: vastus ei tohi lisada uusi arvulisi konkreetseid fakte,
    # mida kontekstis ei ole (nt päevad, summad, numbrid, viitenumbrid).
    def _extract_numeric_tokens(text: str) -> set:
        if not text:
            return set()
        tokens = set()
        for raw in re.findall(r"\d+(?:[.,]\d+)?", text):
            normalized = raw.replace(",", ".").strip()
            if normalized:
                tokens.add(normalized)
        return tokens

    context_numbers = _extract_numeric_tokens(req.context or "")
    response_numbers = _extract_numeric_tokens(req.ai_response or "")
    new_numbers = sorted(x for x in response_numbers if x not in context_numbers)
    if new_numbers:
        return {
            "model": "hard-rule",
            "status": "BLOCKED",
            "reason": f"Vastus lisab kontekstis puuduvaid arvulisi fakte: {', '.join(new_numbers[:5])}.",
            "duration": 0.0,
            "prompt": "",
            "raw_response": "",
        }

    quality_model = (req.quality_model or req.model or "gemma2:2b").strip()
    quality_template = logic_core.PROMPTS.get("POST_CHECK_PROMPT", "{u_input}\\n{main_res}")
    quality_prompt = (
        quality_template
        .replace("{u_input}", req.original_user_input)
        .replace("{context}", (req.context or "").strip() or "Puudub")
        .replace("{main_res}", req.ai_response)
    )
    quality_prompt = quality_prompt.replace(
        "{normalized_query}",
        (req.normalized_query or "").strip() or req.original_user_input,
    )
    quality_start = time.time()
    quality_raw = logic_core.ask_ollama(quality_model, quality_prompt, req.threads, req.timeout)
    quality_duration_ms = round((time.time() - quality_start) * 1000, 2)
    quality_parsed = logic_core.parse_json_res(quality_raw)
    quality_status = str(quality_parsed.get("status", "BLOCKED")).upper()
    quality_reason = quality_parsed.get("reason", "")
    return {
        "model": quality_model,
        "status": quality_status,
        "reason": quality_reason,
        "duration": quality_duration_ms,
        "prompt": quality_prompt,
        "raw_response": quality_raw,
    }


def _run_security_post_check(req: PostCheckRequest):
    def _is_secret_candidate(item: dict) -> bool:
        meta = (item or {}).get("metadata", {}) or {}
        level = str(meta.get("classification_level", "")).strip().lower()
        return level == "secret"

    def _extract_sensitive_ids(candidates: List[dict]) -> set:
        ids = set()
        for item in candidates or []:
            meta = (item or {}).get("metadata", {}) or {}
            for key in ("subject_id", "counterparty_id", "personal_id"):
                value = str(meta.get(key, "")).strip()
                if value:
                    ids.add(value)
        return ids

    def _contains_unmasked_personal_id(text: str, known_ids: set) -> bool:
        if not text:
            return False
        if "***" in text:
            masked_text = text.replace("***", "")
        else:
            masked_text = text

        # 1) Kui teame konkreetseid ID-sid metadatast, kontrollime neid otse.
        for pid in known_ids:
            if pid and pid in masked_text:
                return True

        # 2) Mustrikontroll: "isikukood <id>" või "personal id <id>" tüüpi lekke tuvastus.
        pattern = re.compile(r"(isikukood|personal\s*id)\s*[:#]?\s*([A-Za-z0-9\-]{6,20})", re.IGNORECASE)
        return bool(pattern.search(masked_text))

    candidates = req.sources_returned_raw or []
    ai_response_text = req.ai_response or ""

    # Hard rule 1: secret=false korral ei tohi vastus kasutada salajast konteksti.
    if not bool(req.secret):
        secret_selected = any(
            _is_secret_candidate(item) and bool((item or {}).get("selected", False))
            for item in candidates
        )
        if secret_selected and ai_response_text.strip() != "Esitatud kontekstis info puudub.":
            return {
                "model": "hard-rule",
                "status": "BLOCKED",
                "reason": "Vastus kasutab salajast infot, kuigi secret õigus puudub.",
                "duration": 0.0,
                "prompt": "",
                "raw_response": "",
            }

    # Hard rule 2: allow_personal_data=false korral ei tohi vastus näidata maskeerimata isikuandmeid.
    if not bool(req.allow_personal_data):
        known_ids = _extract_sensitive_ids(candidates)
        if _contains_unmasked_personal_id(ai_response_text, known_ids):
            return {
                "model": "hard-rule",
                "status": "BLOCKED",
                "reason": "Vastus sisaldab maskeerimata isikuandmeid, kuigi see pole lubatud.",
                "duration": 0.0,
                "prompt": "",
                "raw_response": "",
            }

    # Hard rule 3: allow_all_subjects=false ja allowed_subject_ids piirab ligipääsu.
    if not bool(req.allow_all_subjects):
        allowed_subjects = {
            str(x).strip() for x in (req.allowed_subject_ids or []) if str(x).strip()
        }
        if allowed_subjects:
            subject_selected = False
            subject_blocked = False
            for item in candidates:
                if not bool((item or {}).get("selected", False)):
                    continue
                meta = (item or {}).get("metadata", {}) or {}
                subject_id = str(meta.get("subject_id", "")).strip()
                if not subject_id:
                    continue
                subject_selected = True
                if subject_id not in allowed_subjects:
                    subject_blocked = True
                    break
            if subject_selected and subject_blocked:
                return {
                    "model": "hard-rule",
                    "status": "BLOCKED",
                    "reason": "Vastus kasutab subject_id infot, millele kasutajal ligipääs puudub.",
                    "duration": 0.0,
                    "prompt": "",
                    "raw_response": "",
                }

    # Hard rule 4: allowed_tenant_ids piirab ligipääsu tenantile.
    allowed_tenants = {
        str(x).strip() for x in (req.allowed_tenant_ids or []) if str(x).strip()
    }
    if allowed_tenants:
        tenant_selected = False
        tenant_blocked = False
        for item in candidates:
            if not bool((item or {}).get("selected", False)):
                continue
            meta = (item or {}).get("metadata", {}) or {}
            tenant_id = str(meta.get("tenant_id", "")).strip()
            if not tenant_id:
                continue
            tenant_selected = True
            if tenant_id not in allowed_tenants:
                tenant_blocked = True
                break
        if tenant_selected and tenant_blocked:
            return {
                "model": "hard-rule",
                "status": "BLOCKED",
                "reason": "Vastus kasutab tenant_id infot, millele kasutajal ligipääs puudub.",
                "duration": 0.0,
                "prompt": "",
                "raw_response": "",
            }

    security_model = (req.security_model or req.model or "gemma2:2b").strip()
    security_template = logic_core.PROMPTS.get(
        "POST_CHECK_SECURITY_PROMPT",
        "Respond ONLY with JSON: {\"status\":\"ALLOWED\",\"reason\":\"fallback\"}",
    )
    security_prompt = (
        security_template
        .replace("{u_input}", req.original_user_input)
        .replace("{normalized_query}", (req.normalized_query or "").strip() or req.original_user_input)
        .replace("{context}", (req.context or "").strip() or "Puudub")
        .replace("{main_res}", req.ai_response)
        .replace("{secret}", str(bool(req.secret)).lower())
        .replace("{allow_all_subjects}", str(bool(req.allow_all_subjects)).lower())
        .replace("{allow_personal_data}", str(bool(req.allow_personal_data)).lower())
        .replace("{allowed_subject_ids}", json.dumps(req.allowed_subject_ids or [], ensure_ascii=False))
        .replace("{allowed_tenant_ids}", json.dumps(req.allowed_tenant_ids or [], ensure_ascii=False))
        .replace("{sources_returned_raw}", json.dumps(req.sources_returned_raw or [], ensure_ascii=False))
    )
    security_start = time.time()
    security_raw = logic_core.ask_ollama(security_model, security_prompt, req.threads, req.timeout)
    security_duration_ms = round((time.time() - security_start) * 1000, 2)
    security_parsed = logic_core.parse_json_res(security_raw)
    security_status = str(security_parsed.get("status", "BLOCKED")).upper()
    security_reason = security_parsed.get("reason", "")
    return {
        "model": security_model,
        "status": security_status,
        "reason": security_reason,
        "duration": security_duration_ms,
        "prompt": security_prompt,
        "raw_response": security_raw,
    }


@app.post(
    "/post-check-quality",
    tags=["Valideerimine"],
    summary="Post-check sisuline kontroll (soovituslik)",
    description="Soovituslik endpoint post-check sisulise kontrolli jaoks. Kontrollib, kas vastus on konteksti- ja küsimusepõhine.",
)
async def post_check_quality(req: PostCheckRequest, user: str = Depends(authenticate)):
    start_time = time.time()
    try:
        if not req.original_user_input:
            raise HTTPException(status_code=422, detail="Missing required field: original_user_input")
        quality = _run_quality_post_check(req)
        duration = time.time() - start_time
        response_data = {
            "status": quality["status"],
            "reason": quality["reason"],
            "analysis": quality["reason"],
            "duration": round(duration * 1000, 2),
            "check": quality,
        }
        log_api_call("/post-check-quality", 200, duration, user, response_data)
        return response_data
    except Exception as e:
        log_api_call("/post-check-quality", 500, 0, user, {"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/post-check-security",
    tags=["Valideerimine"],
    summary="Post-check turvakontroll (soovituslik)",
    description="Soovituslik endpoint post-check turvakontrolli jaoks. Kontrollib, kas vastus järgib ligipääsu- ja andmereegleid.",
)
async def post_check_security(req: PostCheckRequest, user: str = Depends(authenticate)):
    start_time = time.time()
    try:
        if not req.original_user_input:
            raise HTTPException(status_code=422, detail="Missing required field: original_user_input")
        security = _run_security_post_check(req)
        duration = time.time() - start_time
        response_data = {
            "status": security["status"],
            "reason": security["reason"],
            "analysis": security["reason"],
            "duration": round(duration * 1000, 2),
            "check": security,
        }
        log_api_call("/post-check-security", 200, duration, user, response_data)
        return response_data
    except Exception as e:
        log_api_call("/post-check-security", 500, 0, user, {"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/post-check",
    tags=["Valideerimine"],
    summary="Post-check koondteenus (legacy)",
    description="Legacy koondendpoint, mis käivitab nii sisulise kui turvakontrolli. Uues kasutuses eelista eraldi endpoint'e: /post-check-quality ja /post-check-security.",
)
async def post_check(req: PostCheckRequest, user: str = Depends(authenticate)):
    """
    Koondteenus: käivitab nii sisulise kui turvakontrolli ja tagastab ühise otsuse.
    """
    start_time = time.time()
    start_time_str = time.strftime("%H:%M:%S")
    try:
        if not req.original_user_input:
            raise HTTPException(status_code=422, detail="Missing required field: original_user_input")

        quality = _run_quality_post_check(req)
        security = _run_security_post_check(req)

        final_status = "BLOCKED" if ("BLOCKED" in {quality["status"], security["status"]}) else "ALLOWED"
        reason_parts = []
        if quality["status"] == "BLOCKED" and quality["reason"]:
            reason_parts.append(f"Sisuline kontroll: {quality['reason']}")
        if security["status"] == "BLOCKED" and security["reason"]:
            reason_parts.append(f"Turvakontroll: {security['reason']}")
        final_reason = " | ".join(reason_parts)
        duration = time.time() - start_time

        response_data = {
            "model": req.model,
            "quality_model": quality["model"],
            "security_model": security["model"],
            "start_time": start_time_str,
            "status": final_status,
            "reason": final_reason,
            "analysis": final_reason or ("ALLOWED" if final_status == "ALLOWED" else ""),
            "quality_status": quality["status"],
            "security_status": security["status"],
            "duration": round(duration * 1000, 2),
            "ai_response": req.ai_response,
            "original_user_input": req.original_user_input,
            "normalized_query": req.normalized_query or "",
            "context": req.context or "",
            "checks": {
                "quality": quality,
                "security": security,
            },
        }

        log_api_call("/post-check", 200, duration, user, response_data)
        return response_data
    except Exception as e:
        log_api_call("/post-check", 500, 0, user, {"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/logs", tags=["Süsteem"])
def get_logs(
    user: str = Depends(authenticate), 
    source: str = Query(
        "api",
        enum=[
            "api",
            "ui",
            "test-pre-check",
            "test-post-check",
            "test-post-check-use-cases",
            "test-llm",
            "test-retrieval",
            "test-stability",
            "test-benchmark-embeddings",
            "test-normalizer",
            "prompts-change",
        ],
    ),
    limit: int = Query(50, description="Mitut viimast rida kuvada"),
    start: Optional[str] = Query(None, description="Algusaeg (YYYY-MM-DD HH:MM:SS)"),
    end: Optional[str] = Query(None, description="Lõpuaeg (YYYY-MM-DD HH:MM:SS)")
):
    """Tagastab süsteemi logid filtreeritult."""
    if source == "api":
        path = API_LOG_FILE
    elif source == "ui":
        path = UI_LOG_FILE
    else:
        path = TEST_LOG_FILES.get(source, "")

    if not os.path.exists(path):
        return {"error": f"Logifail asukohas {path} puudub", "data": []}
    
    results = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_content = f.read().strip()

        # Toetame nii JSONL (1 kirje rea kohta), JSON listi kui ka ühte JSON objekti.
        # API ja UI logid on praktikas JSONL; mitu "{...}" rida ei ole üks JSON objekt.
        entries = []
        if raw_content:
            if raw_content.startswith("["):
                try:
                    parsed = json.loads(raw_content)
                    if isinstance(parsed, list):
                        entries = parsed
                except Exception:
                    entries = []
            elif raw_content.startswith("{") and "\n" not in raw_content:
                try:
                    parsed = json.loads(raw_content)
                    if isinstance(parsed, dict):
                        entries = [parsed]
                except Exception:
                    entries = []
            else:
                for line in raw_content.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except Exception:
                        entries.append({
                            "timestamp": "",
                            "status": "ERROR",
                            "endpoint": "log-parse",
                            "error": "Logirida ei olnud korrektne JSON ja jäeti vaates vahele.",
                            "raw_preview": line[:300],
                        })

            if not entries and "\n" in raw_content:
                for line in raw_content.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except Exception:
                        entries.append({
                            "timestamp": "",
                            "status": "ERROR",
                            "endpoint": "log-parse",
                            "error": "Logirida ei olnud korrektne JSON ja jäeti vaates vahele.",
                            "raw_preview": line[:300],
                        })

        for entry in entries[-500:]:
            if not isinstance(entry, dict):
                continue
            ts = (
                entry.get("timestamp")
                or entry.get("run_started_at")
                or entry.get("run_finished_at")
                or ""
            )
            if start and ts < start:
                continue
            if end and ts > end:
                continue
            results.append(entry)
        
        return {
            "source": source, 
            "count": len(results[-limit:]), 
            "data": results[-limit:]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", tags=["Süsteem"])
def health_check():
    """Kontrollib teenuse ja Ollama olekut."""
    ollama_status = "unknown"
    try:
        import subprocess
        res = subprocess.run(["curl", "-s", "http://ollama:11434/api/tags"], capture_output=True, text=True, timeout=5)
        if res.returncode == 0:
            ollama_status = "running"
        else:
            ollama_status = "error"
    except:
        ollama_status = "down"
        
    return {
        "status": "online",
        "api_version": "1.9",
        "ollama_connection": ollama_status,
        "timestamp": ee_now_str()
    }

if __name__ == "__main__":
    import uvicorn
    # Käivitab serveri pordil 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)


