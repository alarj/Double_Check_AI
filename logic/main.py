# --- ver 0.2 -- Alar 09.04, lisatud logimine, eestikeelsem kasutajaliides ja muud veaparandused
# --- ver 0.1 -- Alar 09.04, esialgne versioon Gemini abiga tehtud 
import streamlit as st
import requests
import json
import datetime
import os

# --- KONFIGURATSIOON ---
OLLAMA_URL = "http://ollama:11434/api/generate"
LOG_FILE = "/app/ai_turvakiht.log"
GUARD_MODEL = "phi3:mini"
MAIN_MODEL = "llama3:8b"

# --- UNIVERSAALNE RESSURSSIDE TUVASTAMINE ---
# Tuvastame kõik olemasolevad protsessori tuumad
total_cores = os.cpu_count() or 1
# Kasutame kohe algväärtusena MAKSIMAALSET tuumade arvu
default_threads = total_cores

# --- FUNKTSIOONID ---

def log_to_file(user_input, safety_status, ai_response=""):
    """Salvestab päringu andmed tekstifaili ja tagab õigused."""
    if not os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "w", encoding="utf-8") as f:
                f.write(f"--- Logifail loodud: {datetime.datetime.now()} ---\n")
            os.chmod(LOG_FILE, 0o666)
        except Exception as e:
            print(f"Logifaili loomise viga: {e}")

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
    except Exception as e:
        print(f"Kirjutamise viga: {e}")

def ask_ollama(model, prompt, threads):
    """Saadab päringu Ollama API-le, kasutades kõiki määratud lõime."""
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_thread": threads  # Siia läheb nüüd kogu ressurss
            }
        }
        # Pikk timeout on CPU-põhise täisvõimsuse juures vajalik
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

if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "show_logs" not in st.session_state:
    st.session_state.show_logs = False

# --- KÜLGRIBA ---
with st.sidebar:
    st.title("🛡️ Firma Sise-AI")
    
    st.subheader("Süsteemi ressursid")
    st.success(f"Tuvastatud tuumi: {total_cores} (Kõik kasutusel)")
    # Võimalus käsitsi muuta, kui vaja testimiseks vähendada
    selected_threads = st.number_input("Kasutatavad threads:", min_value=1, max_value=total_cores, value=default_threads)
    
    st.info(f"Turvamoodul: {GUARD_MODEL}\nTöömoodul: {MAIN_MODEL}")
    
    st.markdown("---")
    if st.button("Näita viimaseid päringuid"):
        st.session_state.show_logs = not st.session_state.show_logs

    if st.session_state.show_logs:
        st.subheader("Viimased logid")
        if os.path.exists(LOG_FILE):
            try:
                with open(LOG_FILE, "r", encoding="utf-8") as f:
                    content = f.readlines()
                    st.text_area("Logi sisu:", "".join(content[-25:]), height=400)
            except Exception as e:
                st.error(f"Luku viga: {e}")

# --- PEALEHT ---
st.title("🚀 Firma Sise-AI Turvakiht")

user_input = st.text_input("Sisesta oma küsimus AI-le:", placeholder="Tere!")

if user_input and user_input != st.session_state.last_query:
    st.session_state.last_query = user_input
    
    with st.spinner(f"Töötlen täisvõimsusel ({selected_threads} tuuma)..."):
        # 1. SAMM: Turvakontroll
        guard_prompt = (
            f"Sina oled turvasüsteem. Analüüsi küsimust: '{user_input}'.\n"
            "Vasta ainult üks sõna: LUBATUD või BLOKEERITUD."
        )
        
        safety_result = ask_ollama(GUARD_MODEL, guard_prompt, selected_threads)
        
        if safety_result == "VIGA_TIMEOUT":
            st.error("⌛ Timeout! Isegi täisvõimsusel võttis liiga kaua aega.")
            log_to_file(user_input, "VIGA", "Timeout")
        elif safety_result == "VIGA_ÜHENDUS":
            st.error("🔌 Ollama sideviga.")
        elif "LUBATUD" in safety_result.upper():
            st.success("✅ Turvakontroll läbitud")
            
            # 2. SAMM: Põhivastus
            with st.spinner("Genereerin vastust..."):
                main_response = ask_ollama(MAIN_MODEL, user_input, selected_threads)
                st.markdown(f"**AI vastus:**\n{main_response}")
                log_to_file(user_input, "LUBATUD", main_response)
        else:
            st.error("🚨 PÄRING BLOKEERITUD")
            log_to_file(user_input, "BLOKEERITUD", "Turvamoodul")

elif not user_input:
    st.info("Ootan küsimust...")