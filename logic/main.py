import streamlit as st
import os
import time
import json
import logic_core  # Impordime ühise loogika

# --- KONFIGURATSIOON ---
LOG_FILE = "/app/ai_turvakiht.log"
DEFAULT_GUARD = "gemma2:2b"
DEFAULT_MAIN = "llama3:8b"

total_cores = os.cpu_count() or 1

# --- ABI FUNKTSIOONID ---
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
        options=[
            "Eelkontroll (küsimuse valideerimine) ja põhipäring", 
            "Põhipäring ja järelkontroll (tulemuse valideerimine)", 
            "Eelkontroll, põhipäring, järelkontroll", 
            "Ainult põhipäring"
        ],
        index=0, disabled=is_disabled
    )
    
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
        "timestamp": logic_core.get_ee_time().strftime("%Y-%m-%d %H:%M:%S"),
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

    # 1. Eelkontroll
    if do_pre and not only_main:
        status_box.info("Samm 1: Turvakontroll (Eelanalüüs)...")
        pre_prompt = logic_core.PRE_CHECK_PROMPT.format(u_input=u_input)
        
        log_data["pre_check"].update({"prompt": pre_prompt, "start_time": logic_core.get_ee_time().strftime("%H:%M:%S")})
        res = logic_core.ask_ollama(guard_model_input, pre_prompt, selected_threads, selected_timeout)
        log_data["pre_check"].update({"result": res, "status": "DONE"})
        
        if logic_core.get_first_decision(res) != "LUBATUD":
            is_safe = False
            st.session_state.last_status = "BLOKEERITUD"
            st.session_state.last_response = f"🚨 Sisend blokeeriti turvafiltri poolt.\n(Süsteemi teade: {res})"
            log_data["final_status"] = "BLOCKED_PRE"

    # 2. Põhipäring
    if is_safe:
        status_box.info(f"Samm 2: Genereerin vastust ({main_model_input})...")
        log_data["main_query"].update({"prompt": u_input, "start_time": logic_core.get_ee_time().strftime("%H:%M:%S")})
        main_res = logic_core.ask_ollama(main_model_input, u_input, selected_threads, selected_timeout)
        log_data["main_query"].update({"result": main_res, "status": "DONE"})
        
        if "VIGA" in main_res:
            is_safe = False
            st.session_state.last_status = "VIGA"
            st.session_state.last_response = main_res
            log_data["final_status"] = "ERROR_MAIN"
        else:
            # 3. Järelkontroll
            if do_post and not only_main:
                status_box.info("Samm 3: Turvakontroll (Väljundi analüüs)...")
                post_prompt = logic_core.POST_CHECK_PROMPT.format(main_res=main_res)
                
                log_data["post_check"].update({"prompt": post_prompt, "start_time": logic_core.get_ee_time().strftime("%H:%M:%S")})
                p_res = logic_core.ask_ollama(guard_model_input, post_prompt, selected_threads, selected_timeout)
                log_data["post_check"].update({"result": p_res, "status": "DONE"})
                
                if logic_core.get_first_decision(p_res) != "LUBATUD":
                    is_safe = False
                    st.session_state.last_status = "BLOKEERITUD"
                    st.session_state.last_response = "🚨 Vastus blokeeriti turvakaalutlustel (sisu ei vasta ohutusstandarditele)."
                    log_data["final_status"] = "BLOCKED_POST"
            
            if is_safe:
                st.session_state.last_status = "OK"
                st.session_state.last_response = main_res
                log_data["final_status"] = "OK"

    log_data["total_duration"] = round(time.time() - start_time_perf, 2)
    log_data["end_time"] = logic_core.get_ee_time().strftime("%H:%M:%S")
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
        st.error(st.session_state.last_status)
    else:
        st.warning(st.session_state.last_response)