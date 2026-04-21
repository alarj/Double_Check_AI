from fastapi import FastAPI, Depends, HTTPException, status, Query
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import logic_core
import json
import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from pydantic import BaseModel, Field
from typing import Optional, List

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
    "test-llm": "/testing/llm-test-log.json",
    "test-retrieval": "/testing/retr-test-log.json",
    "test-benchmark-embeddings": "/testing/benchmark_embeddings-log.jsonl",
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
    timeout: Optional[int] = 60
    threads: Optional[int] = 4

class MainQueryRequest(BaseModel):
    user_input: str
    context: str
    model: str = "llama3:8b"
    timeout: Optional[int] = 120
    threads: Optional[int] = 8

class PostCheckRequest(BaseModel):
    ai_response: str
    # Pre-check "user_input" (algne, töötlemata küsimus)
    original_user_input: Optional[str] = Field(None, min_length=1)
    # Pre-check väljund (normaliseeritud küsimus), mille alusel tehti põhiküsimus/RAG
    normalized_query: Optional[str] = ""
    # Kontekst, mille alusel põhipäring LLM-s tehti (RAG kontekst)
    context: Optional[str] = ""
    # Backward compatibility varasema kliendi jaoks
    original_query: Optional[str] = None
    model: str = "gemma2:2b"
    timeout: Optional[int] = 90
    threads: Optional[int] = 4

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
        sys_prompt = logic_core.PROMPTS.get("PRE_CHECK_PROMPT", "{u_input}")
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
        
        response_data = {
            "model": req.model,
            "prompt": full_prompt,
            "start_time": start_time_str,
            "status": parsed_result.get("status", "BLOCKED"),
            "normalized": parsed_result.get("normalized_query", ""),
            "duration": round(duration * 1000, 2),
            "raw_response": result
        }
        
        log_api_call("/pre-check", 200, duration, user, response_data)
        return response_data
    except Exception as e:
        log_api_call("/pre-check", 500, 0, user, {"error": str(e)})
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

@app.post("/post-check", tags=["Valideerimine"])
async def post_check(req: PostCheckRequest, user: str = Depends(authenticate)):
    """
    Kontrollib AI vastuse vastavust algsele küsimusele (hallutsinatsioonide vältimine).
    """
    start_time = time.time()
    start_time_str = time.strftime("%H:%M:%S")
    try:
        # Toetame vanemat kliendilepingut (original_query) ilma 422 errorita.
        if (not req.original_user_input) and req.original_query:
            req.original_user_input = req.original_query
        if not req.original_user_input:
            raise HTTPException(status_code=422, detail="Missing required field: original_user_input (or legacy original_query)")

        # Võtame õige võtmega prompti failist
        sys_prompt = logic_core.PROMPTS.get("POST_CHECK_PROMPT", "{u_input}\n{main_res}")
        # Asendame kohamärgised vastavalt failile prompts.json
        full_prompt = (
            sys_prompt
            .replace("{u_input}", req.original_user_input)
            .replace("{context}", (req.context or "").strip() or "Puudub")
            .replace("{main_res}", req.ai_response)
        )
        # Kui prompt kasutab lisa-kohamärke, täidame ka need (kui olemas)
        full_prompt = full_prompt.replace("{normalized_query}", (req.normalized_query or "").strip() or req.original_user_input)
        
        result = logic_core.ask_ollama(
            req.model,
            full_prompt,
            req.threads,
            req.timeout
        )
        duration = time.time() - start_time
        
        # Parsime Ollama string vastuse reaalseks JSONiks
        parsed_result = logic_core.parse_json_res(result)
        
        response_data = {
            "model": req.model,
            "prompt": full_prompt,
            "start_time": start_time_str,
            "status": parsed_result.get("status", "BLOCKED"),
            "reason": parsed_result.get("reason", ""),
            "analysis": parsed_result.get("analysis", parsed_result.get("reason", "")),
            "duration": round(duration * 1000, 2),
            "raw_response": result,
            "ai_response": req.ai_response,
            "original_user_input": req.original_user_input,
            "normalized_query": req.normalized_query or "",
            "context": req.context or "",
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
            "test-llm",
            "test-retrieval",
            "test-benchmark-embeddings",
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

        # Toetame nii JSONL (1 kirje rea kohta) kui ka JSON list formaati.
        entries = []
        if raw_content:
            if raw_content.startswith("["):
                parsed = json.loads(raw_content)
                if isinstance(parsed, list):
                    entries = parsed
            else:
                for line in raw_content.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    entries.append(json.loads(line))

        for entry in entries[-500:]:
            if not isinstance(entry, dict):
                continue
            ts = entry.get("timestamp", "")
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