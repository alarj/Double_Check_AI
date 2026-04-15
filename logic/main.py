import streamlit as st
import os
import time
import json
import logic_core

LOG_FILE = "/app/ai_turvakiht.log"
DEFAULT_GUARD = "gemma2:2b"
DEFAULT_MAIN = "llama3:8b"
total_cores = os.cpu_count() or 1

def log_json_event(data):
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    except: pass

st.set_page_config(page_title="AI Turvakiht", layout="wide")

if "processing" not in st.session_state: st.session_state.processing = False
if "last_response" not in st.session_state: st.session_state.last_response = None
if "last_status" not in st.session_state: st.session_state.last_status = None
if "show_logs" not in st.session_state: st.session_state.show_logs = False
if "current_query" not in st.session_state: st.session_state.current_query = None

with st.sidebar:
    st.title("🛡️ Firma Sise-AI")
    is_disabled = st.session_state.processing
    selected_threads = st.number_input("Threads:", 1, total_cores, total_cores, disabled=is_disabled)
    selected_timeout = st.number_input("Timeout (sek):", 30, 1200, 360, disabled=is_disabled)
    guard_model_input = st.text_input("Turvamudel:", DEFAULT_GUARD, disabled=is_disabled)
    main_model_input = st.text_input("Põhimudel:", DEFAULT_MAIN, disabled=is_disabled)
    security_option = st.selectbox("Turvalisuse tase:", options=[
        "Eelkontroll (küsimuse valideerimine) ja põhipäring", 
        "Põhipäring ja järelkontroll (tulemuse valideerimine)", 
        "Eelkontroll, põhipäring, järelkontroll", 
        "Ainult põhipäring"
    ], index=0, disabled=is_disabled)
    
    if st.button("Näita/Peida logid", disabled=is_disabled):
        st.session_state.show_logs = not st.session_state.show_logs

st.title("🚀 Firma Sise-AI Turvakiht")

with st.form(key="query_form"):
    user_input = st.text_input("Sisesta küsimus:", disabled=st.session_state.processing)
    if st.form_submit_button("Saada päring") and user_input:
        st.session_state.processing = True
        st.session_state.current_query = user_input
        st.rerun()

if st.session_state.processing and st.session_state.current_query:
    u_input = st.session_state.current_query
    start_time_perf = time.time()
    log_data = {
        "timestamp": logic_core.get_ee_time().strftime("%Y-%m-%d %H:%M:%S"),
        "user_input": u_input, "security_level": security_option,
        "pre_check": {"status": "SKIP"}, "main_query": {"status": "SKIP"},
        "post_check": {"status": "SKIP"}, "final_status": "PENDING"
    }
    
    is_safe = True
    status_box = st.empty()

    # 1. Eelkontroll
    if "Eelkontroll" in security_option:
        status_box.info("Samm 1: Turvakontroll...")
        res = logic_core.ask_ollama(guard_model_input, logic_core.PRE_CHECK_PROMPT.format(u_input=u_input), selected_threads, selected_timeout)
        log_data["pre_check"].update({"result": res, "status": "DONE"})
        if logic_core.get_first_decision(res) != "LUBATUD":
            is_safe = False
            st.session_state.last_status = "BLOKEERITUD"
            st.session_state.last_response = f"🚨 Blokeeritud: {res}"
            log_data["final_status"] = "BLOCKED_PRE"

    # 2. Põhipäring (RAG-iga)
    if is_safe:
        status_box.info("Samm 2: Otsin infot andmebaasist ja genereerin vastust...")
        
        # RAG samm: Leiame asjakohase konteksti ja koostame prompti
        context = logic_core.get_context(u_input)
        rag_prompt = logic_core.RAG_PROMPT.format(context=context, query=u_input)
        
        # Küsime vastust põhimudelilt
        main_res = logic_core.ask_ollama(main_model_input, rag_prompt, selected_threads, selected_timeout)
        log_data["main_query"].update({"result": main_res, "context_used": context, "status": "DONE"})
        
        if "VIGA" in main_res:
            is_safe = False
            st.session_state.last_status = "VIGA"
            st.session_state.last_response = main_res
        else:
            # 3. Järelkontroll
            p_res = "KONTROLLIMATA" 
            if "järelkontroll" in security_option:
                status_box.info("Samm 3: Väljundi analüüs ja turvakontroll...")
                
                # Järelkontrolli prompti koostamine (kasutame nii u_input kui main_res)
                # Süsteem võtab raamistiku prompts.json failist
                p_prompt = logic_core.POST_CHECK_PROMPT.format(u_input=u_input, main_res=main_res)
                
                # Päring kontrollmoodulile
                p_res = logic_core.ask_ollama(guard_model_input, p_prompt, selected_threads, selected_timeout)
                
                decision = logic_core.get_first_decision(p_res)
                log_data["post_check"].update({"result": p_res, "status": "DONE"})
                
                if decision != "LUBATUD":
                    is_safe = False
                    st.session_state.last_status = "BLOKEERITUD"
                    # Näitame kasutajale, et sisu blokeeriti, aga lisame ka kontrolli selgituse
                    st.session_state.last_response = f"🚨 **BLOKEERITUD**\n\n**AI algne vastus:** {main_res}\n\n**Kontroll:** {p_res}"
                    log_data["final_status"] = "BLOCKED_POST"
            
            if is_safe:
                st.session_state.last_status = "OK"
                # KOOSTAME LIITVASTUSE KASUTAJALE (Sisu + Kontrollmooduli arvamus)
                combined_text = f"**AI vastus:** {main_res}\n\n---\n**Kontroll:** {p_res}"
                st.session_state.last_response = combined_text
                log_data["final_status"] = "OK"
                # Salvestame liitvastuse ka logisse
                log_data["main_query"]["combined_result"] = combined_text

    log_data["total_duration"] = round(time.time() - start_time_perf, 2)
    log_json_event(log_data)
    st.session_state.processing = False
    st.session_state.current_query = None
    status_box.empty()
    st.rerun()

if st.session_state.last_response:
    if st.session_state.last_status == "OK": st.success(st.session_state.last_response)
    else: st.warning(st.session_state.last_response)