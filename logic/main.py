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

# Automaatne CPU tuumade tuvastus (universaalne)
total_cores = os.cpu_count() or 1
default_threads = total_cores

# --- FUNKTSIOONID ---

def log_to_file(user_input, safety_status, ai_response=""):
    """Salvestab päringu andmed tekstifaili ja tagab õigused."""
    if not os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "w", encoding="utf-8") as f:
                f.write(f"--- Logifail loodud: {datetime.datetime.now()} ---\n")
            os.chmod(LOG_FILE, 0o666)
        except:
            pass

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    clean_ai_res = ai_response[:200].replace('\n', ' ') + "..." if ai_response else "-"
    
    log_entry = (
        f"[{timestamp}]\n"
        f"KÜSIMUS: {user_input}\n"
        f"STAATUS: {safety_status}\n"
        f"VASTUS: {clean_ai_res}\n"
        f"{'-'*50}\n"
    )

    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_entry)
    except:
        pass

def ask_ollama(model, prompt, threads):
    """Saadab päringu Ollama API-le spetsiifilise veakäsitlusega."""
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_thread": threads
            }
        }
        # Timeout on 300 sekundit (5 minutit), et CPU jõuaks vastata
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
        
        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            return f"VIGA_KOOD_{response.status_code}"
            
    except requests.exceptions.Timeout:
        return "VIGA_TIMEOUT"
    except requests.exceptions.ConnectionError:
        return "VIGA_ÜHENDUS"
    except Exception as e:
        return f"VIGA_MÄÄRAMATA: {str(e)}"

# --- VEEBILEHE SEADISTUS ---
st.set_page_config(page_title="AI Turvakiht", layout="wide")

# Sessiooni haldus (et vastused ei kaoks lehe värskendamisel)
if "last_response" not in st.session_state:
    st.session_state.last_response = None
if "last_status" not in st.session_state:
    st.session_state.last_status = None
if "show_logs" not in st.session_state:
    st.session_state.show_logs = False
if "processing" not in st.session_state:
    st.session_state.processing = False

# --- KÜLGRIBA ---
with st.sidebar:
    st.title("🛡️ Firma Sise-AI")
    st.subheader("Süsteemi ressursid")
    st.info(f"Tuvastatud tuumi: {total_cores}")
    
    # Threads sisend lukustub, kui töö käib
    selected_threads = st.number_input(
        "Threads kasutusel:", 
        min_value=1, 
        max_value=total_cores, 
        value=default_threads,
        disabled=st.session_state.processing
    )
    
    st.markdown("---")
    
    # Logide nupp on lukus, kui AI genereerib (hoiab ära katkestuse)
    if st.button("Näita/Peida logid", disabled=st.session_state.processing):
        st.session_state.show_logs = not st.session_state.show_logs

    if st.session_state.show_logs:
        st.subheader("Viimased logid")
        if os.path.exists(LOG_FILE):
            try:
                with open(LOG_FILE, "r", encoding="utf-8") as f:
                    content = f.readlines()
                    st.text_area("Logi sisu (viimased read):", "".join(content[-25:]), height=400)
            except:
                st.error("Logifaili lugemine ebaõnnestus.")
        else:
            st.write("Logifail puudub.")

# --- PEALEHT ---
st.title("🚀 Firma Sise-AI Turvakiht")

# Vormi kasutamine stabiilsuse tagamiseks
with st.form(key="query_form", clear_on_submit=False):
    user_input = st.text_input("Sisesta oma küsimus AI-le:", placeholder="Kirjuta siia...")
    submit_button = st.form_submit_button(label="Saada päring")

# Konteiner, kuhu vastus ilmub
response_container = st.empty()

if submit_button and user_input:
    # Lülitame sisse töörežiimi (lukustab nupud)
    st.session_state.processing = True
    st.session_state.last_response = None  # Puhastame vana vastuse
    
    with st.spinner(f"Töötlen päringut täisvõimsusel ({selected_threads} tuuma)..."):
        # 1. SAMM: Turvakontroll
        guard_prompt = (
            f"Analüüsi küsimust: '{user_input}'. "
            "Vasta ainult üks sõna: LUBATUD või BLOKEERITUD."
        )
        
        safety_result = ask_ollama(GUARD_MODEL, guard_prompt, selected_threads)
        
        # Vigade käsitlus
        if safety_result == "VIGA_TIMEOUT":
            st.session_state.last_response = "⌛ Serveri vastus viibib (Timeout). Proovi uuesti."
            st.session_state.last_status = "VIGA"
            log_to_file(user_input, "TIMEOUT", "Süsteem oli liiga aeglane")
            
        elif safety_result == "VIGA_ÜHENDUS":
            st.session_state.last_response = "🔌 Sideviga Ollama serveriga."
            st.session_state.last_status = "VIGA"
            
        elif "LUBATUD" in safety_result.upper():
            # 2. SAMM: Tegelik vastus
            main_response = ask_ollama(MAIN_MODEL, user_input, selected_threads)
            
            if "VIGA" in main_response:
                st.session_state.last_response = f"Vastus ebaõnnestus: {main_response}"
                st.session_state.last_status = "VIGA"
            else:
                st.session_state.last_response = main_response
                st.session_state.last_status = "OK"
                log_to_file(user_input, "LUBATUD", main_response)
        
        else:
            # Päring blokeeriti turvamooduli poolt
            st.session_state.last_response = "🚨 PÄRING BLOKEERITUD: Turvamoodul tuvastas ohu siseinfole."
            st.session_state.last_status = "BLOKEERITUD"
            log_to_file(user_input, "BLOKEERITUD", "Turvamooduli otsus")

    # Töö lõpetatud, vabastame nupud ja värskendame lehte
    st.session_state.processing = False
    st.rerun()

# VASTUSE KUVAMINE (isegi pärast lehe uuesti laadimist)
if st.session_state.last_response:
    with response_container.container():
        if st.session_state.last_status == "OK":
            st.success("✅ Vastus valmis")
            st.markdown(st.session_state.last_response)
        elif st.session_state.last_status == "VIGA":
            st.error(st.session_state.last_status + ": " + st.session_state.last_response)
        else:
            st.warning(st.session_state.last_response)