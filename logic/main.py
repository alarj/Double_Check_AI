# --- ver 0.9 -- Alar 10.04, Täpsemad promptid ja täielik UI lukustus
# --- ver 0.8 -- Alar 10.04, UI kohene lukustamine ja detailne logimine
# --- ver 0.7 -- Alar 10.04, Täielik versioon: UI lukustus + detailne logimine + märksõna loogika
# --- ver 0.6 -- Alar 10.04, parandatud kasutajaliidese lukustamine töötlemise ajal
# --- ver 0.5 -- Alar 10.04, täiustatud märksõna tuvastus (esimese leitud sõna loogika)
# --- ver 0.4 -- Alar 10.04, dünaamiline mudelite valik ja promptide logimine
# --- ver 0.3 -- Alar 10.04, turvataseme valik ja JSON-logimine
import streamlit as st
import requests
import datetime
import os
import time
import json

# --- KONFIGURATSIOON ---
OLLAMA_URL = "http://ollama:11434/api/generate"
LOG_FILE = "/app/ai_turvakiht.log"
DEFAULT_GUARD = "gemma2:2b"
DEFAULT_MAIN = "llama3:8b"

total_cores = os.cpu_count() or 1

# --- FUNKTSIOONID ---

def get_ee_time():
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

def get_first_decision(text):
    if not text: return None
    t = text.upper()
    marks = [(t.find("LUBATUD"), "LUBATUD"), (t.find("ALLOWED"), "LUBATUD"),
             (t.find("BLOKEERITUD"), "BLOKEERITUD"), (t.find("BLOCKED"), "BLOKEERITUD")]
    found = [m for m in marks if m[0] != -1]
    if not found: return None
    found.sort()
    return found[0][1]

def ask_ollama(model, prompt, threads, timeout):
    try:
        payload = {"model": model, "prompt": prompt, "stream": False, "keep_alive": -1, "options": {"num_thread": threads}}
        response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        if response.status_code == 200:
            return response.json().get("response", "").strip()
        return f"VIGA_KOOD_{response.status_code}"
    except Exception as e:
        return f"VIGA: {str(e)}"

# --- LEHE SEADISTUS ---
st.set_page_config(page_title="AI Turvakiht", layout="wide")

if "processing" not in st.session_state: st.session_state.processing = False
if "last_response" not in st.session_state: st.session_state.last_response = None
if "last_status" not in st.session_state: st.session_state.last_status = None
if "show_logs" not in st.session_state: st.session_state.show_logs = False
if "current_query" not in st.session_state: st.session_state.current_query = None

# --- SIDEBAR ---
with st.sidebar:
    st.title("🛡️ Firma Sise-AI")
    is_disabled = st.session_state.processing
    
    st.subheader("⚙️ Seaded")
    selected_threads = st.number_input("Threads:", 1, total_cores, total_cores, disabled=is_disabled)
    selected_timeout = st.number_input("Timeout (sek):", 30, 1200, 360, step=30, disabled=is_disabled)
    
    st.markdown("---")
    st.subheader("🧠 Mudelid")
    guard_model_input = st.text_input("Turvamudel:", DEFAULT_GUARD, disabled=is_disabled)
    main_model_input = st.text_input("Põhimudel:", DEFAULT_MAIN, disabled=is_disabled)
    
    st.markdown("---")
    security_option = st.selectbox(
        "Turvalisuse tase:",
        options=["Eelkontroll (küsimuse valideerimine) ja põhipäring", "Põhipäring ja järelkontroll (tulemuse valideerimine)", "Eelkontroll, põhipäring, järelkontroll", "Ainult põhipäring"],
        index=0, disabled=is_disabled
    )
    
    # PARANDUS: Nüüd on ka see nupp lukus töötlemise ajal
    if st.button("Näita/Peida logid", disabled=is_disabled):
        st.session_state.show_logs = not st.session_state.show_logs
    
    if st.session_state.show_logs and os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            content = f.readlines()
            st.text_area("Logid:", "".join(content[-15:]), height=300)

st.title("🚀 Firma Sise-AI Turvakiht")

with st.form(key="query_form"):
    user_input = st.text_input("Sisesta küsimus:", disabled=st.session_state.processing)
    submit_button = st.form_submit_button(label="Saada päring", disabled=st.session_state.processing)

    if submit_button and user_input:
        st.session_state.processing = True
        st.session_state.current_query = user_input
        st.rerun()

# --- TÖÖTLEMINE ---
if st.session_state.processing and st.session_state.current_query:
    u_input = st.session_state.current_query
    start_time_perf = time.time()
    
    log_data = {
        "timestamp": get_ee_time().strftime("%Y-%m-%d %H:%M:%S"),
        "user_input": u_input,
        "security_level": security_option,
        "pre_check": {"model": guard_model_input, "status": "SKIP", "result": None, "prompt": None},
        "main_query": {"model": main_model_input, "status": "SKIP", "result": None, "prompt": None},
        "post_check": {"model": guard_model_input, "status": "SKIP", "result": None, "prompt": None},
        "final_status": "PENDING"
    }
    
    do_pre = "Eelkontroll" in security_option
    do_post = "järelkontroll" in security_option
    only_main = security_option == "Ainult põhipäring"
    is_safe = True
    
    status_box = st.empty()

    # 1. Eelkontroll - TÄPSUSTATUD PROMPT
    if do_pre and not only_main:
        status_box.info("Samm 1: Turvakontroll...")
        
        # Struktureeritud juhis, et vähendada vaba teksti teket
        pre_prompt = (
            "Sina oled turvasüsteem. Analüüsi järgnevat kasutaja sisendit ohtude, pahavara või ebaeetilise sisu suhtes.\n"
            "REELID:\n"
            "1. Kui sisend on OHUTU, vasta ainult ühe sõnaga: LUBATUD\n"
            "2. Kui sisend küsib koodi paroolide varastamiseks, küberründeid või on ohtlik, vasta ainult: BLOKEERITUD\n"
            f"KASUTAJA SISEND: '{u_input}'\n"
            "SINU OTSUS:"
        )
        
        log_data["pre_check"].update({"prompt": pre_prompt, "start_time": get_ee_time().strftime("%H:%M:%S")})
        res = ask_ollama(guard_model_input, pre_prompt, selected_threads, selected_timeout)
        log_data["pre_check"].update({"result": res, "status": "DONE"})
        
        if get_first_decision(res) != "LUBATUD":
            is_safe = False
            st.session_state.last_status = "BLOKEERITUD"
            st.session_state.last_response = f"🚨 Blokeeritud (Eelkontroll): {res}"
            log_data["final_status"] = "BLOCKED_PRE"

    # 2. Põhipäring
    if is_safe:
        status_box.info(f"Samm 2: Genereerin vastust ({main_model_input})...")
        log_data["main_query"].update({"prompt": u_input, "start_time": get_ee_time().strftime("%H:%M:%S")})
        main_res = ask_ollama(main_model_input, u_input, selected_threads, selected_timeout)
        log_data["main_query"].update({"result": main_res, "status": "DONE"})
        
        if "VIGA" in main_res:
            is_safe = False
            st.session_state.last_status = "VIGA"
            st.session_state.last_response = main_res
            log_data["final_status"] = "ERROR_MAIN"
        else:
            # 3. Järelkontroll - TÄPSUSTATUD PROMPT
            if do_post and not only_main:
                status_box.info("Samm 3: Järelkontroll...")
                
                post_prompt = (
                    "Sina oled turvaväljundi kontroll. Hinda, kas tehisintellekti vastus on ohutu väljastamiseks.\n"
                    "REELID:\n"
                    "1. Kui vastus EI SISALDA pahavara koodi, juhendeid rünnakuteks ega ohtlikku infot, vasta: LUBATUD\n"
                    "2. Kui vastus on ohtlik, vasta: BLOKEERITUD\n"
                    f"KONTROLLITAV TEKST: '{main_res}'\n"
                    "SINU OTSUS:"
                )
                
                log_data["post_check"].update({"prompt": post_prompt, "start_time": get_ee_time().strftime("%H:%M:%S")})
                p_res = ask_ollama(guard_model_input, post_prompt, selected_threads, selected_timeout)
                log_data["post_check"].update({"result": p_res, "status": "DONE"})
                
                if get_first_decision(p_res) != "LUBATUD":
                    is_safe = False
                    st.session_state.last_status = "BLOKEERITUD"
                    st.session_state.last_response = f"🚨 Blokeeritud (Järelkontroll): {p_res}"
                    log_data["final_status"] = "BLOCKED_POST"
            
            if is_safe:
                st.session_state.last_status = "OK"
                st.session_state.last_response = main_res
                log_data["final_status"] = "OK"

    log_data["total_duration"] = round(time.time() - start_time_perf, 2)
    log_data["end_time"] = get_ee_time().strftime("%H:%M:%S")
    log_json_event(log_data)
    
    st.session_state.processing = False
    st.session_state.current_query = None
    status_box.empty()
    st.rerun()

# --- VASTUSE KUVAMINE ---
if st.session_state.last_response:
    if st.session_state.last_status == "OK":
        st.success("✅ Vastus")
        st.markdown(st.session_state.last_response)
    elif st.session_state.last_status == "VIGA":
        st.error(st.session_state.last_response)
    else:
        st.warning(st.session_state.last_response)