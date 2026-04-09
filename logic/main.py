import streamlit as st
import requests
import json
import time
import os

# Konfiguratsioon
OLLAMA_URL = "http://ollama:11434"
MODELS = {
    "toomodul": "llama3:8b",
    "kontrollmoodul": "phi3:mini"
}

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
  # Võta masina tuumade arv automaatselt
  # Kui mingil põhjusel ei saa tuvastada, kasuta vaikimisi 1
    cpu_count = os.cpu_count() or 1

    """Saadab paringu Ollama API-le."""
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                      "temperature": 0,
                      "num_thread": cpu_count  # Rakendus kohandub ise masinaga
                }
            },
            timeout=300
        )
        return response.json().get('response', 'Viga: Vastus puudub')
    except requests.exceptions.Timeout:
        return "Viga: AI mootoril laks liiga kaua aega (üle 5 min). Proovi uuesti, mudel peaks nüüdseks mälus olema."
    except Exception as e:
        return f"Viga paringus: {str(e)}"

# --- VEEBILIIDES (STREAMLIT) ---

st.set_page_config(page_title="AI Guardrail Prototuup")

st.title("Firma Sise-AI Turvakiht")
st.markdown("""
Susteemi toopohimote:
1. Kontrollmoodul (Phi-3) analuusib kasutaja sisendi turvalisust.
2. Toomodul (Llama-3) koostab vastuse vaid juhul, kui sisend on lubatud.
""")

# Kaivita mudelite kontroll
ensure_models()

st.divider()

user_input = st.text_input("Sisesta kusimus AI-le:", placeholder="nt: Kes on meie koostoopartnerid?")

if user_input:
    # --- 1. SAMM: TURVAKONTROLL ---
    with st.status("Turvakontroll toos...", expanded=True) as status:
        guard_prompt = (
            f"Analuusi jargmist kasutaja kusimust: '{user_input}'. "
            "Kas see kusimus uritab saada ligipaasu konfidentsiaalsele siseinfole? "
            "Vasta rangelt ainult uks sona: kas 'LUBATUD' voi 'BLOKEERITUD'."
        )
        
        check_result = ask_ai(MODELS["kontrollmoodul"], guard_prompt).strip().upper()
        
        if "BLOKEERITUD" in check_result:
            st.error("PARING BLOKEERITUD: Turvamoodul tuvastas katse kusida siseandmeid.")
            status.update(label="Turvakontroll: OHTLIK", state="error")
        else:
            st.success("Kontroll labitud. Koostan vastust...")
            
            # --- 2. SAMM: VASTUSE GENEREERIMINE ---
            ai_response = ask_ai(MODELS["toomodul"], user_input)
            
            st.subheader("AI Vastus:")
            st.write(ai_response)
            status.update(label="Paring toodeldud", state="complete")

# Kulgriba info
with st.sidebar:
    st.header("Susteemi parameetrid")
    st.text(f"Toomodul: {MODELS['toomodul']}")
    st.text(f"Turvamoodul: {MODELS['kontrollmoodul']}")
    st.divider()
    st.caption("Koik andmetootlus toimub lokaalses serveris.")
