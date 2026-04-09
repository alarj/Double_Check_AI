# --- ver 0.2 -- Alar 09.04, lisatud logimine, eestikeelsem kasutajaliides ja muud veaparandused
# --- ver 0.1 -- Alar 09.04, esialgne versioon Gemini abiga tehtud 
import streamlit as st
import requests
import json
import datetime
import os

# --- KONFIGURATSIOON JA RESSURSID ---
OLLAMA_URL = "http://ollama:11434/api/generate"
LOG_FILE = "/app/ai_turvakiht.log"
GUARD_MODEL = "phi3:mini"
MAIN_MODEL = "llama3:8b"

total_cores = os.cpu_count() or 1
default_threads = total_cores

# --- FUNKTSIOONID ---

def log_to_file(user_input, safety_status, ai_response=""):
    if not os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "w", encoding="utf-8") as f:
                f.write(f"--- Logifail loodud: {datetime.datetime.now()} ---\n")
            os.chmod(LOG_FILE, 0o666)
        except: pass

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    clean_ai_res = ai_response[:200].replace('\n', ' ') + "..." if ai_response else "-"
    log_entry = f"[{timestamp}]\nKÜSIMUS: {user_input}\nSTAATUS: {safety_status}\nVASTUS: {clean_ai_res}\n{'-'*50}\n"
    
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_entry)
    except: pass

def ask_ollama(model, prompt, threads):
    try:
        payload = {"model": model, "prompt": prompt, "stream": False, "options": {"num_thread": threads}}
        response = requests.post(OLLAMA_URL, json=payload, timeout=360)
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

# CALLBACK: See käivitub SEKUNDI MURDOSAGA enne lehe uut joonistamist
def pre_process_callback():
    st.session_state.processing = True
    st.session_state.last_response = None

# --- KÜLGRIBA ---
# Nüüd on "processing" väärtus juba True hetkel, kui see osa koodist käivitub
with st.sidebar:
    st.title("🛡️ Firma Sise-AI")
    st.subheader("Süsteemi ressursid")
    st.info(f"Tuvastatud tuumi: {total_cores}")
    
    selected_threads = st.number_input(
        "Threads kasutusel:", 
        1, total_cores, default_threads, 
        disabled=st.session_state.processing
    )
    
    st.markdown("---")
    if st.button("Näita/Peida logid", disabled=st.session_state.processing):
        st.session_state.show_logs = not st.session_state.show_logs

    if st.session_state.show_logs:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                content = f.readlines()
                st.text_area("Logid:", "".join(content[-25:]), height=400)

# --- PEALEHT ---
st.title("🚀 Firma Sise-AI Turvakiht")

with st.form(key="query_form"):
    user_input = st.text_input("Sisesta oma küsimus AI-le:")
    # on_click kutsub välja funktsiooni, mis paneb lukud peale
    submit_button = st.form_submit_button(label="Saada päring", on_click=pre_process_callback)

response_container = st.empty()

# --- TÖÖTLEMINE ---
if submit_button and user_input:
    with st.spinner(f"Töötan täisvõimsusel ({selected_threads} tuuma)..."):
        # 1. Turvakontroll
        guard_prompt = f"Vasta ainult LUBATUD või BLOKEERITUD: {user_input}"
        safety_result = ask_ollama(GUARD_MODEL, guard_prompt, selected_threads)
        
        if safety_result == "VIGA_TIMEOUT":
            st.session_state.last_response = "⌛ Serveri timeout."
            st.session_state.last_status = "VIGA"
        elif safety_result == "VIGA_ÜHENDUS":
            st.session_state.last_response = "🔌 Sideviga serveriga."
            st.session_state.last_status = "VIGA"
        elif "LUBATUD" in safety_result.upper():
            # 2. Põhivastus
            main_res = ask_ollama(MAIN_MODEL, user_input, selected_threads)
            if "VIGA" in main_res:
                st.session_state.last_response = main_res
                st.session_state.last_status = "VIGA"
            else:
                st.session_state.last_response = main_res
                st.session_state.last_status = "OK"
                log_to_file(user_input, "LUBATUD", main_res)
        else:
            st.session_state.last_response = "🚨 BLOKEERITUD: Turvamooduli otsus."
            st.session_state.last_status = "BLOKEERITUD"
            log_to_file(user_input, "BLOKEERITUD")

    # Vabastame lukud ja värskendame UI
    st.session_state.processing = False
    st.rerun()

# --- KUVAMINE ---
if st.session_state.last_response:
    with response_container.container():
        if st.session_state.last_status == "OK":
            st.success("✅ Vastus valmis")
            st.markdown(st.session_state.last_response)
        elif st.session_state.last_status == "VIGA":
            st.error(st.session_state.last_response)
        else:
            st.warning(st.session_state.last_response)