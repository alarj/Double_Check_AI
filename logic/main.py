# --- ver 0.2 -- Alar 09.04, lisatud logimine, eestikeelsem kasutajaliides ja muud veaparandused
# --- ver 0.1 -- Alar 09.04, esialgne versioon Gemini abiga tehtud 
import streamlit as st
import requests
import datetime
import os
import time

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

def log_event(event_type, message):
    """Logib sündmuse Eesti ajatempliga."""
    timestamp = get_ee_time().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] [{event_type}] {message}\n"
    
    if not os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "w", encoding="utf-8") as f:
                f.write(f"--- Logifail loodud: {get_ee_time()} (EEST) ---\n")
            os.chmod(LOG_FILE, 0o666)
        except: pass
    
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_entry)
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

# Algväärtustame sessiooni muutujad
if "last_response" not in st.session_state: st.session_state.last_response = None
if "last_status" not in st.session_state: st.session_state.last_status = None
if "show_logs" not in st.session_state: st.session_state.show_logs = False
if "processing" not in st.session_state: st.session_state.processing = False
# VAIKIMISI 360 SEKUNDIT
if "current_timeout" not in st.session_state: st.session_state.current_timeout = 360

def start_thinking():
    st.session_state.processing = True
    st.session_state.last_response = None

# --- KÜLGRIBA ---
with st.sidebar:
    st.title("🛡️ Firma Sise-AI")
    st.info(f"Tuvastatud tuumi: {total_cores}")
    
    selected_threads = st.number_input(
        "Threads kasutusel:", 
        1, total_cores, default_threads, 
        disabled=st.session_state.processing
    )
    
    selected_timeout = st.number_input(
        "Päringu timeout (sek):", 
        30, 1200, st.session_state.current_timeout, 
        step=30,
        disabled=st.session_state.processing
    )
    # Salvestame valiku sessiooni
    st.session_state.current_timeout = selected_timeout
    
    # --- UUS: Turvalisuse taseme valik ---
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
                st.text_area("Logid (EEST):", "".join(content[-50:]), height=400)

st.title("🚀 Firma Sise-AI Turvakiht")

with st.form(key="query_form"):
    user_input = st.text_input("Sisesta küsimus AI-le:")
    submit_button = st.form_submit_button(label="Saada päring", on_click=start_thinking)

status_placeholder = st.empty()

# --- TÖÖTLEMINE ---
if submit_button and user_input:
    log_event("START", f"Sisend: {user_input[:40]} | Timeout: {selected_timeout}s | Tase: {security_option}")
    start_time = time.time()
    
    # Loeme valikust välja, millised sammud tuleb läbida
    do_pre = "Eelkontroll" in security_option
    do_post = "järelkontroll" in security_option
    only_main = security_option == "Ainult põhipäring"
    
    safety_result = "LUBATUD (Vahele jäetud)" # Vaikimisi lubatud, kui eelkontrolli ei tehta
    
    # 1. SAMM: Turvakontroll (Eelkontroll)
    if do_pre and not only_main:
        with status_placeholder.container():
            with st.spinner(f"Samm 1: Turvakontroll ({selected_timeout}s max)..."):
                log_event("GUARD_PRE", f"Käivitan eelkontrolli ({GUARD_MODEL})")
                safety_result = ask_ollama(GUARD_MODEL, f"Vasta ainult LUBATUD või BLOKEERITUD: {user_input}", selected_threads, selected_timeout)
        
    if safety_result == "VIGA_TIMEOUT":
        log_event("VIGA", f"Turvakontroll aegus ({selected_timeout}s)")
        st.session_state.last_response = f"⌛ Turvakontroll aegus. Proovi timeouti tõsta (hetkel {selected_timeout}s)."
        st.session_state.last_status = "VIGA"
            
    elif "LUBATUD" in safety_result.upper():
        if do_pre and not only_main:
            log_event("GUARD_PRE", "Otsus: LUBATUD")
        
        # 2. SAMM: Põhivastus
        with status_placeholder.container():
            with st.spinner(f"Samm 2: Genereerin vastust ({MAIN_MODEL})..."):
                log_event("MAIN", f"Käivitan põhimudeli ({MAIN_MODEL})")
                main_res = ask_ollama(MAIN_MODEL, user_input, selected_threads, selected_timeout)
        
        if "VIGA" in main_res:
            log_event("VIGA", f"Põhimudeli tõrge: {main_res}")
            st.session_state.last_response = f"Serveri viga: {main_res}"
            st.session_state.last_status = "VIGA"
        else:
            # 3. SAMM: Järelkontroll
            post_safety_result = "LUBATUD (Vahele jäetud)"
            if do_post and not only_main:
                with status_placeholder.container():
                    with st.spinner(f"Samm 3: Järelkontroll ({selected_timeout}s max)..."):
                        log_event("GUARD_POST", f"Käivitan järelkontrolli ({GUARD_MODEL})")
                        post_safety_result = ask_ollama(GUARD_MODEL, f"Hinda kas see tehisintellekti vastus on ohutu. Vasta ainult LUBATUD või BLOKEERITUD: {main_res}", selected_threads, selected_timeout)
            
            if post_safety_result == "VIGA_TIMEOUT":
                log_event("VIGA", f"Järelkontroll aegus ({selected_timeout}s)")
                st.session_state.last_response = f"⌛ Järelkontroll aegus. Proovi timeouti tõsta (hetkel {selected_timeout}s)."
                st.session_state.last_status = "VIGA"
            elif "LUBATUD" in post_safety_result.upper():
                if do_post and not only_main:
                    log_event("GUARD_POST", "Otsus: LUBATUD")
                
                duration = round(time.time() - start_time, 2)
                log_event("FINISH", f"Vastus valmis {duration}s jooksul.")
                st.session_state.last_response = main_res
                st.session_state.last_status = "OK"
            else:
                log_event("GUARD_POST", "Otsus: BLOKEERITUD")
                st.session_state.last_response = "🚨 Põhimudeli genereeritud vastus blokeeriti järelkontrollis turvakaalutlustel."
                st.session_state.last_status = "BLOKEERITUD"
    else:
        log_event("GUARD_PRE", "Otsus: BLOKEERITUD")
        st.session_state.last_response = "🚨 See päring on turvakaalutlustel blokeeritud."
        st.session_state.last_status = "BLOKEERITUD"

    log_event("INFO", "Sessioon lõppenud\n" + "-"*40)
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