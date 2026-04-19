from fastapi import FastAPI, Depends, HTTPException, status, Query
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import logic_core
import json
import os
import time
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
    context: Optional[str] = ""
    model: str = "deepseek-r1:8b"
    timeout: Optional[int] = 120
    threads: Optional[int] = 8

class PostCheckRequest(BaseModel):
    ai_response: str
    original_query: str
    model: str = "gemma2:2b"
    timeout: Optional[int] = 90
    threads: Optional[int] = 4

# --- LOGIMINE ---
def log_api_call(endpoint: str, status_code: int, duration: float, user: str, extra_data: dict = None):
    """Logib API päringu info faili."""
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
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
        # Võtame õige võtmega prompti failist
        sys_prompt = logic_core.PROMPTS.get("RAG_PROMPT", "{context}\n{query}")
        # Asendame konteksti ja küsimuse kohamärgised
        full_prompt = sys_prompt.replace("{context}", req.context or "Puudub").replace("{query}", req.user_input)
        
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
            "response": result,
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
        # Võtame õige võtmega prompti failist
        sys_prompt = logic_core.PROMPTS.get("POST_CHECK_PROMPT", "{u_input}\n{main_res}")
        # Asendame kohamärgised vastavalt failile prompts.json
        full_prompt = sys_prompt.replace("{u_input}", req.original_query).replace("{context}", "Puudub").replace("{main_res}", req.ai_response)
        
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
            "duration": round(duration * 1000, 2),
            "raw_response": result
        }
        
        log_api_call("/post-check", 200, duration, user, response_data)
        return response_data
    except Exception as e:
        log_api_call("/post-check", 500, 0, user, {"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/logs", tags=["Süsteem"])
def get_logs(
    user: str = Depends(authenticate), 
    source: str = Query("api", enum=["api", "ui"]), 
    limit: int = Query(50, description="Mitut viimast rida kuvada"),
    start: Optional[str] = Query(None, description="Algusaeg (YYYY-MM-DD HH:MM:SS)"),
    end: Optional[str] = Query(None, description="Lõpuaeg (YYYY-MM-DD HH:MM:SS)")
):
    """Tagastab süsteemi logid filtreeritult."""
    path = API_LOG_FILE if source == "api" else UI_LOG_FILE
    if not os.path.exists(path):
        return {"error": f"Logifail asukohas {path} puudub", "data": []}
    
    results = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines[-500:]:
                try:
                    entry = json.loads(line)
                    ts = entry.get("timestamp", "")
                    
                    if start and ts < start: continue
                    if end and ts > end: continue
                    
                    results.append(entry)
                except:
                    continue
        
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
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

if __name__ == "__main__":
    import uvicorn
    # Käivitab serveri pordil 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)