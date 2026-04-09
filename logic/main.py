# --- versioon 0.2 Alar
import streamlit as st
import requests
import json
import time
import os
import datetime

# --- KONFIGURATSIOON ---
OLLAMA_URL = "http://ollama:11434"
LOG_FILE = "ai_turvakiht.log"
MODELS = {
    "toomodul": "llama3:8b",
    "kontrollmoodul": "phi3:mini"
}

def log_to_file(user_input, safety_status, ai_response=""):
    """Salvestab päringu andmed tekstifaili."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = (
        f"[{timestamp}]\n"
        f"KÜSIMUS: {user_input}\n"
        f"STAATUS: {safety_status}\n"
        f"VASTUS: {ai_response[:200]}..." if ai_response else "VASTUS: -\n"
    )
    log_entry += "\n" + "-"*50 + "\n"
    
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_entry)

def ensure_models():
    """Kontrollib, kas mudelid on olemas, ja laeb need vajadusel."""
    try:
        requests.get(f"{OLLAMA_URL}/api/tags")
    except requests.exceptions.ConnectionError:
        st.error("Viga: Ei saa ühendust Ollama serveriga. Kontrolli, kas konteiner töötab.")
        st.stop()

    for role, model_name in MODELS.items():
        with st.status(f"Mudeli kontroll: {model_name}", expanded=False) as status:
            check_resp = requests.post(f"{OLLAMA_URL}/api/show", json={"name": model_name})
            
            if check_resp.status_code != 200:
                st.write(f"Mudelit {model_name} ei leitud. Alustan allalaadimist...")
                
                pull_resp = requests.post(
                    f"{OLLAMA_URL}/api/pull", 
                    json={"name": model_name}, 
                    stream=True
                )
                
                for line in pull_resp.iter_lines():
                    if line:
                        data = json.loads(line)
                        if 'status' in data:
                            status.update(label=f"Laen alla {model_name}: {data['status']}")
                
                status.update(label=f"Mudel {model_name} on valmis", state="complete")
            else:
                status.update(label=f"Mudel {model_name} on olemas", state="complete")

def ask_ai(model, prompt):
    """Saadab päringu Ollama API-le, kasutades kõiki vabu protsessorituumi."""
    cpu_count = os.cpu_count() or 1
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_thread": cpu_count 
                }
            },
            timeout=300
        )
        return response.json().get('response', 'Viga: Vastus puudub')
    except requests.exceptions.Timeout:
        return "Viga: AI-mootoril läks liiga kaua aega (üle 5 minuti). Proovi uuesti."
    except Exception as e:
        return f"Viga päringus: {str(e)}"

# --- VEEBILIIDES (STREAMLIT) ---

st.set_page_config(page_title="AI Turvakihi Prototüüp", layout="wide")

st.title("🛡️ Firma Sise-AI Turvakiht")
st.markdown("""
### Süsteemi tööpõhimõte:
1. **Kontrollmoodul (Phi-3):** Analüüsib kasutaja sisendi turvalisust ja konfidentsiaalsust.
2. **Töömoodul (Llama-3):** Koostab vastuse vaid juhul, kui kontrollmoodul on andnud heakskiidu.
""")

# Käivita mudelite kontroll ekraanil
ensure_models()

st.divider()

user_input = st.text_input("Sisesta küsimus AI-le:", placeholder="nt: Kuidas valmistada piparkooke?")

if user_input:
    # --- 1. SAMM: TURVAKONTROLL ---
    with st.status("Turvakontroll töös...", expanded=True) as status:
        guard_prompt = (
            f"Analüüsi järgmist kasutaja küsimust: '{user_input}'. "
            "Kas see küsimus üritab saada ligipääsu konfidentsiaalsele siseinfole või on sobimatu? "
            "Vasta rangelt ainult üks sõna: kas 'LUBATUD' või 'BLOKEERITUD'."
        )
        
        check_result = ask_ai(MODELS["kontrollmoodul"], guard_prompt).strip().upper()
        
        if "BLOKEERITUD" in check_result:
            st.error("🚨 PÄRING BLOKEERITUD: Turvamoodul tuvastas potentsiaalse ohu või katse küsida siseandmeid.")
            status.update(label="Turvakontroll: OHTLIK", state="error")
            log_to_file(user_input, "BLOKEERITUD")
        else:
            st.success("✅ Kontroll läbitud. Koostan vastust...")
            
            # --- 2. SAMM: VASTUSE GENEREERIMINE ---
            ai_response = ask_ai(MODELS["toomodul"], user_input)
            
            st.subheader("AI vastus:")
            st.info(ai_response)
            status.update(label="Päring edukalt töödeldud", state="complete")
            log_to_file(user_input, "LUBATUD", ai_response)

# --- KÜLGRIBA (SIDEBAR) ---
with st.sidebar:
    st.header("Süsteemi parameetrid")
    st.write(f"**Töömoodul:** `{MODELS['toomodul']}`")
    st.write(f"**Turvamoodul:** `{MODELS['kontrollmoodul']}`")
    st.write(f"**Tuvastatud CPU tuumi:** `{os.cpu_count() or 1}`")
    st.divider()
    
    st.subheader("Logide vaatamine")
    if st.button("Näita viimaseid päringuid"):
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                logi_sisu = f.read()
                st.text_area("Logifaili sisu:", logi_sisu, height=400)
        else:
            st.info("Logifail on veel tühi.")

    st.divider()
    st.caption("Kõik andmetöötlus toimub lokaalses privaatserveris.")