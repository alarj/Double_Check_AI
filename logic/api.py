from fastapi import FastAPI, Depends, HTTPException, status, Query
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import logic_core
import json
import os
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Double_Check_AI REST API", version="1.0", docs_url="/swagger")
security = HTTPBasic()

API_LOG_FILE = "/app/api_access.log"
UI_LOG_FILE = "/app/ai_turvakiht.log"

# --- AUTH ---
def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != "admin" or credentials.password != "parool":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Vale kasutaja või parool")
    return credentials.username

# --- MUDELID ---
class InputRequest(BaseModel):
    user_input: str
    model: str = "gemma2:2b"

class PostCheckRequest(BaseModel):
    user_input: str
    main_response: str
    model: str = "gemma2:2b"

# --- LOGI FUNKTSIOON ---
def log_api_event(action, detail, result, model=None):
    """Logib API sündmuse JSON struktuuris ilma pikkuse piiranguta."""
    log_data = {
        "timestamp": logic_core.get_ee_time().strftime("%Y-%m-%d %H:%M:%S"),
        "action": action,
        "detail": detail,
        "result": result, # Salvestab täispika vastuse
        "model": model,
        "source": "api_rest"
    }
    try:
        with open(API_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Logimisviga: {e}")

# --- ENDPOINTID ---

@app.post("/pre-check", dependencies=[Depends(authenticate)])
def pre_check(req: InputRequest):
    prompt = logic_core.PRE_CHECK_PROMPT.format(u_input=req.user_input)
    res = logic_core.ask_ollama(req.model, prompt, threads=4, timeout=120)
    decision = logic_core.get_first_decision(res) or "BLOKEERITUD"
    log_api_event("PRE_CHECK", req.user_input, decision, req.model)
    return {"decision": decision, "raw_response": res}

@app.post("/main-query", dependencies=[Depends(authenticate)])
def main_query(req: InputRequest):
    res = logic_core.ask_ollama(req.model, req.user_input, threads=4, timeout=300)
    log_api_event("MAIN_QUERY", req.user_input, res, req.model)
    return {"response": res}

@app.post("/post-check", dependencies=[Depends(authenticate)])
def post_check(req: PostCheckRequest):
    prompt = logic_core.POST_CHECK_PROMPT.format(main_res=req.main_response)
    res = logic_core.ask_ollama(req.model, prompt, threads=4, timeout=120)
    decision = logic_core.get_first_decision(res) or "BLOKEERITUD"
    log_api_event("POST_CHECK", req.main_response, decision, req.model)
    return {"decision": decision, "raw_response": res}

@app.get("/logs", dependencies=[Depends(authenticate)])
def get_logs(
    source: str = Query("api", description="Logide allikas: 'api' või 'ui'"), 
    start: str = Query(None, description="Algusaeg (YYYY-MM-DD HH:MM:SS)"), 
    end: str = Query(None, description="Lõpuaeg (YYYY-MM-DD HH:MM:SS)")
):
    path = API_LOG_FILE if source == "api" else UI_LOG_FILE
    if not os.path.exists(path): 
        return {"error": f"Logifail {source} puudub"}
    
    results = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                entry = json.loads(line)
                log_ts = entry.get("timestamp", "")
                if (not start or log_ts >= start) and (not end or log_ts <= end):
                    results.append(entry)
    except Exception as e:
        return {"error": str(e)}
    return results