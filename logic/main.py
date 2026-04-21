import streamlit as st
import os
import time
import json
import subprocess
import base64
import urllib.request
import urllib.parse
import logic_core

# --- KONFIGURATSIOON ---
# Logi asukoht ja promptide fail
LOG_FILE = "/app/ai_turvakiht.log"
PROMPTS_FILE = "/app/prompts.json"
PROMPTS_CHANGE_LOG_FILE = "/logic/prompts_change_log.json"
API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8000")
API_USER = os.getenv("API_USER", "admin")
API_PASSWORD = os.getenv("API_PASSWORD", "parool")
DEFAULT_GUARD = "gemma2:2b"
DEFAULT_MAIN = "llama3:8b"
total_cores = os.cpu_count() or 1

def log_json_event(data):
    """Kirjutab sündmuse logifaili JSON formaadis."""
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"VIGA LOGIMISEL: {e}")

def append_prompt_change_log(old_prompts, new_prompts):
    """Lisab promptide muudatuse auditikirje JSON-faili."""
    entry = {
        "timestamp": logic_core.get_ee_time().strftime("%Y-%m-%d %H:%M:%S"),
        "old_prompts": old_prompts,
        "new_prompts": new_prompts,
    }
    try:
        existing = []
        if os.path.exists(PROMPTS_CHANGE_LOG_FILE):
            with open(PROMPTS_CHANGE_LOG_FILE, "r", encoding="utf-8") as f:
                raw = f.read().strip()
                if raw:
                    parsed = json.loads(raw)
                    if isinstance(parsed, list):
                        existing = parsed
        existing.append(entry)
        with open(PROMPTS_CHANGE_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.error(f"Prompti muutuse logimine ebaõnnestus: {e}")

def fetch_logs_via_api(source, limit=50):
    """Toob logid REST API /logs endpointist."""
    try:
        auth_token = base64.b64encode(f"{API_USER}:{API_PASSWORD}".encode("utf-8")).decode("utf-8")
        query = urllib.parse.urlencode({"source": source, "limit": limit})
        url = f"{API_BASE_URL}/logs?{query}"
        req = urllib.request.Request(url, headers={"Authorization": f"Basic {auth_token}"})
        with urllib.request.urlopen(req, timeout=10) as res:
            data = json.loads(res.read().decode("utf-8"))
            return data.get("data", []), None
    except Exception as e:
        return [], str(e)

def detect_git_branch():
    """Tagastab git haru nime kui olemas."""
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            cwd="/app",
        ).strip()
        return branch or "teadmata"
    except Exception:
        return "teadmata"

# --- UI SEADISTAMINE ---
st.set_page_config(page_title="Sinu nutikas AI assistent", layout="wide", page_icon="⚖️")

# Session state algväärtustamine
if "processing" not in st.session_state: st.session_state.processing = False
if "last_response" not in st.session_state: st.session_state.last_response = None
if "last_post_analysis" not in st.session_state: st.session_state.last_post_analysis = None
if "last_status" not in st.session_state: st.session_state.last_status = None
if "show_logs" not in st.session_state: st.session_state.show_logs = False
if "edit_prompts" not in st.session_state: st.session_state.edit_prompts = False
if "current_query" not in st.session_state: st.session_state.current_query = None
if "log_source" not in st.session_state: st.session_state.log_source = "ui"
if "status_messages" not in st.session_state: st.session_state.status_messages = []
if "last_elapsed_sec" not in st.session_state: st.session_state.last_elapsed_sec = 0.0

# --- SIDEBAR ---
with st.sidebar:
    st.title("🛡️ Seaduste AI")
    is_disabled = st.session_state.processing
    build_time = os.getenv("BUILD_TIME", "teadmata")
    build_branch = os.getenv("BUILD_BRANCH", detect_git_branch())
    
    st.subheader("Serveri sätted")
    st.caption(f"Build aeg: {build_time}")
    st.caption(f"Git haru: {build_branch}")
    selected_threads = st.number_input("Lõimi (threads):", 1, total_cores, total_cores, disabled=is_disabled)
    selected_timeout = st.number_input("Timeout (sek):", 30, 1200, 360, disabled=is_disabled)
    
    st.divider()
    st.subheader("Mudelid")
    guard_model_input = st.text_input("Turva/Normaliseerija:", DEFAULT_GUARD, disabled=is_disabled)
    main_model_input = st.text_input("Põhimudel (RAG):", DEFAULT_MAIN, disabled=is_disabled)
    
    security_option = st.selectbox("Turvalisuse tase:", options=[
        "Eelkontroll (küsimuse valideerimine) ja põhipäring", 
        "Põhipäring ja järelkontroll (tulemuse valideerimine)", 
        "Täiskontroll (Eelkontroll, põhipäring, järelkontroll)", 
        "Ainult põhipäring"
    ], index=2, disabled=is_disabled)
    
    st.divider()
    if st.button("📋 Näita/Peida logid", disabled=is_disabled):
        st.session_state.show_logs = not st.session_state.show_logs
        st.rerun()

    if st.button("📝 Muuda prompte", disabled=is_disabled):
        st.session_state.edit_prompts = not st.session_state.edit_prompts
        st.rerun()
    
    timer_placeholder = st.empty()

# --- PEALEHT ---
st.title("🚀 Sinu nutikas AI assistent")
st.caption("Süsteem kasutab bge-m3 embeddinguid ja struktuurset riigihangete andmebaasi.")
status_placeholder = st.empty()

# --- PÄRINGU VORM ---
with st.form(key="query_form", clear_on_submit=False):
    user_input = st.text_input("Sisesta oma küsimus riigihangete kohta:", placeholder="nt: mis on riigihange?", disabled=st.session_state.processing)
    submit_button = st.form_submit_button("Saada päring", disabled=st.session_state.processing)

if submit_button and user_input:
    st.session_state.processing = True
    st.session_state.current_query = user_input
    st.session_state.edit_prompts = False 
    st.session_state.status_messages = []
    st.session_state.last_elapsed_sec = 0.0
    st.rerun()

# --- PROMPTIDE MUUTMISE VAADE ---
if st.session_state.edit_prompts and not st.session_state.processing:
    st.divider()
    st.subheader("📝 Süsteemi promptide haldus")
    try:
        with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
            current_prompts = json.load(f)
    except Exception:
        current_prompts = logic_core.PROMPTS
    prompts_text = json.dumps(current_prompts, indent=2, ensure_ascii=False)
    st.json(current_prompts)
    
    with st.form("prompt_form"):
        new_prompts_json = st.text_area("Muuda prompts.json sisu:", value=prompts_text, height=400)
        col1, col2 = st.columns([1, 5])
        with col1:
            save_clicked = st.form_submit_button("Salvesta")
        with col2:
            cancel_clicked = st.form_submit_button("Sulge")

        if save_clicked:
            try:
                parsed_json = json.loads(new_prompts_json)
                old_prompts = current_prompts
                with open(PROMPTS_FILE, "w", encoding="utf-8") as f:
                    json.dump(parsed_json, f, indent=2, ensure_ascii=False)
                append_prompt_change_log(old_prompts, parsed_json)
                st.success("Promptid salvestatud!")
                logic_core.PROMPTS = parsed_json
                time.sleep(1)
                st.session_state.edit_prompts = False
                st.rerun()
            except Exception as e:
                st.error(f"Vigane JSON! Kontrolli süntaksit: {e}")
        
        if cancel_clicked:
            st.session_state.edit_prompts = False
            st.rerun()
    st.divider()

# --- TÖÖTLUSLOOGIKA ---
if st.session_state.processing and st.session_state.current_query:
    u_input = st.session_state.current_query
    start_time_total = time.time()
    
    # Algatame logiobjekti vastavalt näidisele
    log_data = {
        "timestamp": logic_core.get_ee_time().strftime("%Y-%m-%d %H:%M:%S"),
        "user_input": u_input,
        "security_level": security_option,
        "steps": {},
        "final_status": "PENDING",
        "total_duration": 0
    }
    
    active_query = u_input
    is_safe = True
    main_answer = ""
    post_analysis = ""
    context_found = False
    fetched_context = ""

    def update_ui(msg):
        elapsed = round(time.time() - start_time_total, 1)
        st.session_state.last_elapsed_sec = elapsed
        timer_placeholder.metric("Kestus", f"{elapsed} sek")
        st.session_state.status_messages.append(msg)
        with status_placeholder.container():
            st.subheader("🔎 Töötlemise sammud")
            for status_msg in st.session_state.status_messages:
                st.info(status_msg)

    try:
        if any(x in security_option for x in ["Eelkontroll", "Täiskontroll"]):
            step_start = time.time()
            update_ui("🔍 Samm 1/4: Päringu valideerimine...")
            pre_p_template = logic_core.PROMPTS.get("PRE_CHECK_PROMPT", "")
            pre_p = pre_p_template.replace("{u_input}", u_input)
            pre_res = logic_core.ask_ollama(guard_model_input, pre_p, selected_threads, selected_timeout)
            status, normalized = logic_core.parse_pre_check(pre_res)
            pre_duration = round(time.time() - step_start, 2)
            update_ui(f"✅ Samm 1/4 valmis ({pre_duration} sek)")

            log_data["steps"]["pre_check"] = {
                "model": guard_model_input,
                "prompt": pre_p,
                "start_time": logic_core.get_ee_time().strftime("%H:%M:%S"),
                "status": status,
                "normalized": normalized if normalized else u_input,
                "duration": round(pre_duration * 1000, 2),
                "raw_response": pre_res
            }

            if status == "ALLOWED":
                if normalized:
                    active_query = normalized
            else:
                is_safe = False
                main_answer = f"🚨 **Päring blokeeritud turvafiltri poolt.**\n\nSelgitus: {pre_res}"
                log_data["final_status"] = "BLOCKED"

        if is_safe:
            update_ui(f"📚 Samm 2/4: Otsin konteksti | päring: {active_query}")
            ctx_start = time.time()
            fetched_context = logic_core.get_context(active_query)
            context_found = len(fetched_context.strip()) > 0
            ctx_duration = round(time.time() - ctx_start, 2)
            update_ui(f"✅ Samm 2/4 valmis ({ctx_duration} sek) | päring: {active_query}")

            log_data["steps"]["context_fetch"] = {
                "found": context_found,
                "duration": ctx_duration
            }

            if not context_found:
                is_safe = False
                main_answer = "Esitatud kontekstis info puudub."
                log_data["final_status"] = "NO_CONTEXT"
            else:
                update_ui(f"🧠 Samm 3/4: Genereerin vastust ({main_model_input}) | päring: {active_query}")
                rag_p_template = logic_core.PROMPTS.get("RAG_PROMPT", "")
                rag_p = rag_p_template.replace("{context}", fetched_context).replace("{query}", active_query)

                step_start_main = time.time()
                main_answer = logic_core.ask_ollama(main_model_input, rag_p, selected_threads, selected_timeout)
                main_duration = round(time.time() - step_start_main, 2)
                update_ui(f"✅ Samm 3/4 valmis ({main_duration} sek) | päring: {active_query}")

                log_data["steps"]["main_query"] = {
                    "model": main_model_input,
                    "prompt": rag_p,
                    "start_time": logic_core.get_ee_time().strftime("%H:%M:%S"),
                    "context_used": fetched_context,
                    "result": main_answer,
                    "duration": main_duration
                }

                if any(x in security_option for x in ["järelkontroll", "Täiskontroll"]):
                    update_ui("🛡️ Samm 4/4: Teen vastuse kvaliteedikontrolli...")
                    post_p_template = logic_core.PROMPTS.get("POST_CHECK_PROMPT", "")
                    post_p = post_p_template.replace("{u_input}", u_input).replace("{context}", fetched_context).replace("{main_res}", main_answer)

                    step_start_post = time.time()
                    post_res = logic_core.ask_ollama(guard_model_input, post_p, selected_threads, selected_timeout)
                    post_data = logic_core.parse_json_res(post_res)
                    p_status = post_data.get("status", "ALLOWED")
                    post_analysis = post_data.get("analysis", post_res)
                    post_duration = round(time.time() - step_start_post, 2)
                    update_ui(f"✅ Samm 4/4 valmis ({post_duration} sek)")

                    log_data["steps"]["post_check"] = {
                        "model": guard_model_input,
                        "prompt": post_p,
                        "start_time": logic_core.get_ee_time().strftime("%H:%M:%S"),
                        "status": p_status,
                        "duration": post_duration,
                        "analysis": post_analysis,
                        "raw_response": post_res
                    }

                    if p_status == "BLOCKED":
                        is_safe = False
                        main_answer = "🚨 **Vastus blokeeriti järelkontrolli poolt.**"
                        log_data["final_status"] = "BLOCKED"

        st.session_state.last_response = main_answer
        st.session_state.last_post_analysis = post_analysis
        st.session_state.last_status = "OK" if is_safe else "BLOCKED"

    except Exception as e:
        st.error(f"Kriitiline viga: {e}")
        log_data["error"] = str(e)
        st.session_state.last_status = "ERROR"
    
    finally:
        log_data["total_duration"] = round(time.time() - start_time_total, 2)
        st.session_state.last_elapsed_sec = log_data["total_duration"]
        timer_placeholder.metric("Kestus", f"{round(st.session_state.last_elapsed_sec, 1)} sek")
        if is_safe and log_data["final_status"] == "PENDING":
            log_data["final_status"] = "OK"
        
        log_json_event(log_data)
        st.session_state.processing = False
        st.rerun()

if st.session_state.processing:
    timer_placeholder.metric("Kestus", f"{round(st.session_state.last_elapsed_sec, 1)} sek")
else:
    if st.session_state.last_elapsed_sec > 0:
        timer_placeholder.metric("Kestus", f"{round(st.session_state.last_elapsed_sec, 1)} sek")

if st.session_state.status_messages:
    with status_placeholder.container():
        st.subheader("🔎 Töötlemise sammud")
        for msg in st.session_state.status_messages:
            st.info(msg)
else:
    status_placeholder.empty()

# --- VÄLJUNDI KUVAMINE ---
if st.session_state.last_response:
    st.divider()
    if st.session_state.last_status == "OK":
        st.subheader("💡 Tehisintellekti vastus:")
        st.markdown(st.session_state.last_response)
        if st.session_state.last_post_analysis:
            with st.expander("🛡️ Kvaliteedikontrolli selgitus", expanded=False):
                st.info(st.session_state.last_post_analysis)
    else:
        st.warning(st.session_state.last_response)
        if st.session_state.last_post_analysis:
            with st.expander("🛡️ Miks vastus blokeeriti?", expanded=True):
                st.error(st.session_state.last_post_analysis)

# --- LOGID UI-S ---
if st.session_state.show_logs:
    st.divider()
    st.subheader("📋 Viimased tegevused")
    log_source = st.selectbox(
        "Logi allikas:",
        options=[
            "ui",
            "api",
            "test-pre-check",
            "test-post-check",
            "test-llm",
            "test-retrieval",
            "test-benchmark-embeddings",
            "prompts-change",
        ],
        index=[
            "ui",
            "api",
            "test-pre-check",
            "test-post-check",
            "test-llm",
            "test-retrieval",
            "test-benchmark-embeddings",
            "prompts-change",
        ].index(st.session_state.log_source) if st.session_state.log_source in [
            "ui",
            "api",
            "test-pre-check",
            "test-post-check",
            "test-llm",
            "test-retrieval",
            "test-benchmark-embeddings",
            "prompts-change",
        ] else 0
    )
    st.session_state.log_source = log_source

    logs, err = fetch_logs_via_api(log_source, limit=50)
    if err:
        st.error(f"Logide päring API kaudu ebaõnnestus: {err}")
    elif not logs:
        st.info("Valitud allikas ei tagastanud logikirjeid.")
    else:
        for entry in reversed(logs):
            if not isinstance(entry, dict):
                st.text(str(entry))
                continue
            status_value = entry.get("final_status") or entry.get("status")
            if status_value in ("OK", "ALLOWED"):
                mumm = "🟢"
            elif status_value in ("BLOCKED", "ERROR"):
                mumm = "🔴"
            else:
                mumm = "🟡"
            title_text = entry.get("user_input") or entry.get("endpoint") or "logikirje"
            label = f"{mumm} {entry.get('timestamp', '---')} | {str(title_text)[:40]}"
            with st.expander(label):
                st.json(entry)