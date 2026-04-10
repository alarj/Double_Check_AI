# --- ver 0.3 -- Alar 10.04, turvataseme valik, JSON-põhine logimine ja täisandmete talletamine
# --- ver 0.2 -- Alar 09.04, lisatud logimine, eestikeelsem kasutajaliides ja muud veaparandused
import streamlit as st
import requests
import datetime
import os
import time
import json  # Lisatud JSON tugi

# --- KONFIGURATSIOON ---
OLLAMA_URL = "http://ollama:11434/api/generate"
LOG_FILE = "/app/ai_turvakiht.log"
GUARD_MODEL = "phi3:mini"
MAIN_MODEL = "llama3:8b"

total_cores = os.cpu_count() or 1
default_threads = total_cores

# --- FUNKTSIOONID ---

def get_ee_time():
    """Tagastab Eesti aja (UTC+3)."""
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=3)))

def log_json_event(data):
    if not os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "w", encoding="utf-8") as f: pass
            os.chmod(LOG_FILE, 0o666)
        except: pass
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    except: pass

def ask_ollama(model, prompt, threads, timeout):
    """Saadab päringu kasutades määratud timeouti."""
    try:
        payload = {"model": model, "prompt": prompt, "stream": False, "options": {"num_thread": threads}}
        response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        if response.status_code == 200:
            return response.json().get("response", "").strip()
        return f"VIGA_KOOD_{response.status_code}"
    except requests.exceptions.Timeout:
        return "VIGA_TIMEOUT"
    except:
        return "VIGA_ÜHENDUS"

# --- VEEBILEHE SEADISTUS ---
st.set_page_config(page_title="AI Turvakiht", layout="wide")

if "last_response" not in st.session_state: st.session_state.last_response = None
if "last_status" not in st.session_state: st.session_state.last_status = None
if "show_logs" not in st.session_state: st.session_state.show_logs = False
if "processing" not in st.session_state: st.session_state.processing = False
if "current_timeout" not in st.session_state: st.session_state.current_timeout = 360

def start_thinking():
    st.session_state.processing = True
    st.session_state.last_response = None

# --- KÜLGRIBA ---
with st.sidebar:
    st.title("🛡️ Firma Sise-AI")
    st.info(f"Tuvastatud tuumi: {total_cores}")
    
    selected_threads = st.number_input("Threads kasutusel:", 1, total_cores, default_threads, disabled=st.session_state.processing)
    selected_timeout = st.number_input("Päringu timeout (sek):", 30, 1200, st.session_state.current_timeout, step=30, disabled=st.session_state.processing)
    st.session_state.current_timeout = selected_timeout
    
    st.markdown("---")
    security_option = st.selectbox(
        "Vali turvalisuse tase:",
        options=[
            "Eelkontroll (küsimuse valideerimine) ja põhipäring",
            "Põhipäring ja järelkontroll (tulemuse valideerimine)",
            "Eelkontroll, põhipäring, järelkontroll",
            "Ainult põhipäring"
        ],
        index=0,
        disabled=st.session_state.processing
    )
    
    st.markdown("---")
    if st.button("Näita/Peida logid", disabled=st.session_state.processing):
        st.session_state.show_logs = not st.session_state.show_logs

    if st.session_state.show_logs:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                content = f.readlines()
                st.text_area("Logid (JSON):", "".join(content[-20:]), height=400)

st.title("🚀 Firma Sise-AI Turvakiht")

with st.form(key="query_form"):
    user_input = st.text_input("Sisesta küsimus AI-le:")
    submit_button = st.form_submit_button(label="Saada päring", on_click=start_thinking)

status_placeholder = st.empty()


# --- TÖÖTLEMINE ---
if submit_button and user_input:
    # 1. FIKSEERIME REAALSE ALGUSE
    query_start_dt = get_ee_time().strftime("%Y-%m-%d %H:%M:%S")
    start_time_perf = time.time()
    
    # log_event("START", f"Sisend: {user_input[:40]} | Tase: {security_option}")
    
    # Initsialiseerime logiandmed staatilise algusajaga
    log_data = {
        "timestamp": query_start_dt,
        "user_input": user_input,
        "security_level": security_option,
        "timeout": selected_timeout,
        "threads": selected_threads,
        "pre_check": {"model": GUARD_MODEL, "status": "SKIP", "result": None, "start_time": None},
        "main_query": {"model": MAIN_MODEL, "status": "SKIP", "result": None, "start_time": None},
        "post_check": {"model": GUARD_MODEL, "status": "SKIP", "result": None, "start_time": None},
        "total_duration": 0,
        "end_time": None,
        "final_status": "PENDING"
    }
    
    do_pre = "Eelkontroll" in security_option
    do_post = "järelkontroll" in security_option
    only_main = security_option == "Ainult põhipäring"
    
    safety_result = "LUBATUD"
    
    # 1. SAMM: Eelkontroll
    if do_pre and not only_main:
        with status_placeholder.container():
            with st.spinner(f"Samm 1: Turvakontroll ({selected_timeout}s)..."):
                log_data["pre_check"]["start_time"] = get_ee_time().strftime("%Y-%m-%d %H:%M:%S")
                safety_result = ask_ollama(GUARD_MODEL, f"Vasta ainult LUBATUD või BLOKEERITUD: {user_input}", selected_threads, selected_timeout)
                log_data["pre_check"].update({"status": "DONE", "result": safety_result})
        
    if safety_result == "VIGA_TIMEOUT":
        log_data["final_status"] = "TIMEOUT_PRE"
        st.session_state.last_response = f"⌛ Turvakontroll aegus ({selected_timeout}s)."
        st.session_state.last_status = "VIGA"
            
    elif "LUBATUD" in safety_result.upper() or not do_pre or only_main:
        # 2. SAMM: Põhivastus
        with status_placeholder.container():
            with st.spinner(f"Samm 2: Genereerin vastust ({MAIN_MODEL})..."):
                log_data["main_query"]["start_time"] = get_ee_time().strftime("%Y-%m-%d %H:%M:%S")
                main_res = ask_ollama(MAIN_MODEL, user_input, selected_threads, selected_timeout)
                log_data["main_query"].update({"status": "DONE", "result": main_res})
        
        if "VIGA" in main_res:
            log_data["final_status"] = f"ERROR_MAIN_{main_res}"
            st.session_state.last_response = f"Serveri viga: {main_res}"
            st.session_state.last_status = "VIGA"
        else:
            # 3. SAMM: Järelkontroll
            post_safety_result = "LUBATUD"
            if do_post and not only_main:
                with status_placeholder.container():
                    with st.spinner(f"Samm 3: Järelkontroll ({selected_timeout}s)..."):
                        log_data["post_check"]["start_time"] = get_ee_time().strftime("%Y-%m-%d %H:%M:%S")
                        post_safety_result = ask_ollama(GUARD_MODEL, f"Hinda kas see vastus on ohutu. Vasta ainult LUBATUD või BLOKEERITUD: {main_res}", selected_threads, selected_timeout)
                        log_data["post_check"].update({"status": "DONE", "result": post_safety_result})
            
            if post_safety_result == "VIGA_TIMEOUT":
                log_data["final_status"] = "TIMEOUT_POST"
                st.session_state.last_response = "⌛ Järelkontroll aegus."
                st.session_state.last_status = "VIGA"
            elif "LUBATUD" in post_safety_result.upper() or not do_post or only_main:
                log_data["final_status"] = "OK"
                st.session_state.last_response = main_res
                st.session_state.last_status = "OK"
            else:
                log_data["final_status"] = "BLOCKED_POST"
                st.session_state.last_response = "🚨 Vastus blokeeriti järelkontrollis."
                st.session_state.last_status = "BLOKEERITUD"
    else:
        log_data["final_status"] = "BLOCKED_PRE"
        st.session_state.last_response = "🚨 Päring blokeeritud."
        st.session_state.last_status = "BLOKEERITUD"

    # LÕPETAMINE: Arvutame kestuse ja salvestame lõpuaja
    log_data["total_duration"] = round(time.time() - start_time_perf, 2)
    log_data["end_time"] = get_ee_time().strftime("%Y-%m-%d %H:%M:%S")

    # Salvestame JSON logi
    log_json_event(log_data)
    
    # VABASTAME PROTSESSI
    st.session_state.processing = False
    st.rerun()

# --- VASTUSE KUVAMINE ---
if st.session_state.last_response:
    if st.session_state.last_status == "OK":
        st.success("✅ Vastus valmis")
        st.markdown(st.session_state.last_response)
    elif st.session_state.last_status == "VIGA":
        st.error(st.session_state.last_response)
    else:
        st.warning(st.session_state.last_response)