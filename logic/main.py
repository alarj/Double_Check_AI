import streamlit as st
import os
import time
import json
import logic_core

# --- KONFIGURATSIOON ---
# Logi asukoht ja promptide fail
LOG_FILE = "/app/ai_turvakiht.log"
PROMPTS_FILE = "/app/prompts.json"
DEFAULT_GUARD = "gemma2:2b"
DEFAULT_MAIN = "llama3:8b"
total_cores = os.cpu_count() or 1

def log_json_event(data):
    """Kirjutab sündmuse logifaili JSON formaadis."""
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"VIGA LOGIMISEL: {e}")

# --- UI SEADISTAMINE ---
st.set_page_config(page_title="Sinu nutikas AI assistent", layout="wide", page_icon="⚖️")

# Session state algväärtustamine
if "processing" not in st.session_state: st.session_state.processing = False
if "last_response" not in st.session_state: st.session_state.last_response = None
if "last_post_analysis" not in st.session_state: st.session_state.last_post_analysis = None
if "last_status" not in st.session_state: st.session_state.last_status = None
if "show_logs" not in st.session_state: st.session_state.show_logs = False
if "edit_prompts" not in st.session_state: st.session_state.edit_prompts = False
if "current_query" not in st.session_state: st.session_state.current_query = None

# --- SIDEBAR ---
with st.sidebar:
    st.title("🛡️ Seaduste AI")
    is_disabled = st.session_state.processing
    
    st.subheader("Serveri sätted")
    selected_threads = st.number_input("Lõimi (threads):", 1, total_cores, total_cores, disabled=is_disabled)
    selected_timeout = st.number_input("Timeout (sek):", 30, 1200, 360, disabled=is_disabled)
    
    st.divider()
    st.subheader("Mudelid")
    guard_model_input = st.text_input("Turva/Normaliseerija:", DEFAULT_GUARD, disabled=is_disabled)
    main_model_input = st.text_input("Põhimudel (RAG):", DEFAULT_MAIN, disabled=is_disabled)
    
    security_option = st.selectbox("Turvalisuse tase:", options=[
        "Eelkontroll (küsimuse valideerimine) ja põhipäring", 
        "Põhipäring ja järelkontroll (tulemuse valideerimine)", 
        "Täiskontroll (Eelkontroll, põhipäring, järelkontroll)", 
        "Ainult põhipäring"
    ], index=2, disabled=is_disabled)
    
    st.divider()
    if st.button("📋 Näita/Peida logid", disabled=is_disabled):
        st.session_state.show_logs = not st.session_state.show_logs
        st.rerun()

    if st.button("📝 Muuda prompte", disabled=is_disabled):
        st.session_state.edit_prompts = not st.session_state.edit_prompts
        st.rerun()
    
    timer_placeholder = st.empty()

# --- PEALEHT ---
st.title("🚀 Sinu nutikas AI assistent")
st.caption("Süsteem kasutab bge-m3 embeddinguid ja struktuurset riigihangete andmebaasi.")

# --- PROMPTIDE MUUTMISE VAADE ---
if st.session_state.edit_prompts and not st.session_state.processing:
    st.divider()
    st.subheader("📝 Süsteemi promptide haldus")
    current_prompts = logic_core.PROMPTS
    prompts_text = json.dumps(current_prompts, indent=2, ensure_ascii=False)
    
    with st.form("prompt_form"):
        new_prompts_json = st.text_area("Muuda prompts.json sisu:", value=prompts_text, height=400)
        col1, col2 = st.columns([1, 5])
        with col1:
            save_clicked = st.form_submit_button("Salvesta")
        with col2:
            cancel_clicked = st.form_submit_button("Sulge")

        if save_clicked:
            try:
                parsed_json = json.loads(new_prompts_json)
                with open(PROMPTS_FILE, "w", encoding="utf-8") as f:
                    json.dump(parsed_json, f, indent=2, ensure_ascii=False)
                st.success("Promptid salvestatud!")
                logic_core.PROMPTS = parsed_json
                time.sleep(1)
                st.session_state.edit_prompts = False
                st.rerun()
            except Exception as e:
                st.error(f"Vigane JSON! Kontrolli süntaksit: {e}")
        
        if cancel_clicked:
            st.session_state.edit_prompts = False
            st.rerun()
    st.divider()

# --- PÄRINGU VORM ---
with st.form(key="query_form", clear_on_submit=False):
    user_input = st.text_input("Sisesta oma küsimus riigihangete kohta:", placeholder="nt: mis on riigihange?", disabled=st.session_state.processing)
    submit_button = st.form_submit_button("Saada päring")

if submit_button and user_input:
    st.session_state.processing = True
    st.session_state.current_query = user_input
    st.session_state.edit_prompts = False 
    st.rerun()

# --- TÖÖTLUSLOOGIKA ---
if st.session_state.processing and st.session_state.current_query:
    u_input = st.session_state.current_query
    start_time_total = time.time()
    status_container = st.empty()
    
    # Algatame logiobjekti vastavalt näidisele
    log_data = {
        "timestamp": logic_core.get_ee_time().strftime("%Y-%m-%d %H:%M:%S"),
        "user_input": u_input,
        "security_level": security_option,
        "steps": {},
        "final_status": "PENDING",
        "total_duration": 0
    }
    
    active_query = u_input
    is_safe = True
    main_answer = ""
    post_analysis = ""
    context_found = False
    fetched_context = ""

    def update_ui(msg):
        elapsed = round(time.time() - start_time_total, 1)
        timer_placeholder.metric("Kestus", f"{elapsed} sek")
        status_container.info(msg)

    try:
        # 1. SAMM: EELKONTROLL
        if any(x in security_option for x in ["Eelkontroll", "Täiskontroll"]):
            update_ui("🔍 Samm 1/4: Päringu valideerimine...")
            pre_p_template = logic_core.PROMPTS.get("PRE_CHECK_PROMPT", "")
            pre_p = pre_p_template.replace("{u_input}", u_input)
            
            step_start = time.time()
            pre_res = logic_core.ask_ollama(guard_model_input, pre_p, selected_threads, selected_timeout)
            status, normalized = logic_core.parse_pre_check(pre_res)
            
            log_data["steps"]["pre_check"] = {
                "model": guard_model_input,
                "prompt": pre_p,
                "start_time": logic_core.get_ee_time().strftime("%H:%M:%S"),
                "status": status,
                "normalized": normalized if normalized else u_input,
                "duration": round((time.time() - step_start) * 1000, 2), # ms täpsus nagu näidises (või sekundid)
                "raw_response": pre_res
            }

            if status == "ALLOWED":
                if normalized: active_query = normalized
            else:
                is_safe = False
                main_answer = f"🚨 **Päring blokeeritud turvafiltri poolt.**\n\nSelgitus: {pre_res}"
                log_data["final_status"] = "BLOCKED"

        # 2. SAMM: KONTEKSTI OTSING (RAG)
        if is_safe:
            update_ui(f"📚 Samm 2/4: Otsin konteksti: {active_query}")
            ctx_start = time.time()
            fetched_context = logic_core.get_context(active_query)
            context_found = len(fetched_context.strip()) > 0
            
            log_data["steps"]["context_fetch"] = {
                "found": context_found,
                "duration": round(time.time() - ctx_start, 2)
            }
            
            if not context_found:
                is_safe = False
                main_answer = "Esitatud kontekstis info puudub."
                log_data["final_status"] = "NO_CONTEXT"
            else:
                # 3. SAMM: PÕHIPÄRING
                update_ui(f"🧠 Samm 3/4: Genereerin vastust ({main_model_input})...")
                rag_p_template = logic_core.PROMPTS.get("RAG_PROMPT", "")
                rag_p = rag_p_template.replace("{context}", fetched_context).replace("{query}", active_query)
                
                step_start_main = time.time()
                main_answer = logic_core.ask_ollama(main_model_input, rag_p, selected_threads, selected_timeout)
                
                log_data["steps"]["main_query"] = {
                    "model": main_model_input,
                    "prompt": rag_p,
                    "start_time": logic_core.get_ee_time().strftime("%H:%M:%S"),
                    "context_used": fetched_context,
                    "result": main_answer,
                    "duration": round(time.time() - step_start_main, 2)
                }
                
                # 4. SAMM: JÄRELKONTROLL
                if any(x in security_option for x in ["järelkontroll", "Täiskontroll"]):
                    update_ui("🛡️ Samm 4/4: Teen vastuse kvaliteedikontrolli...")
                    post_p_template = logic_core.PROMPTS.get("POST_CHECK_PROMPT", "")
                    post_p = post_p_template.replace("{u_input}", u_input).replace("{context}", fetched_context).replace("{main_res}", main_answer)
                    
                    step_start_post = time.time()
                    post_res = logic_core.ask_ollama(guard_model_input, post_p, selected_threads, selected_timeout)
                    post_data = logic_core.parse_json_res(post_res)
                    p_status = post_data.get("status", "ALLOWED") # Vaikimisi lubame kui json katki
                    post_analysis = post_data.get("analysis", post_res)
                    
                    log_data["steps"]["post_check"] = {
                        "model": guard_model_input,
                        "prompt": post_p,
                        "start_time": logic_core.get_ee_time().strftime("%H:%M:%S"),
                        "status": p_status,
                        "duration": round(time.time() - step_start_post, 2),
                        "analysis": post_analysis,
                        "raw_response": post_res
                    }
                    
                    if p_status == "BLOCKED":
                        is_safe = False
                        main_answer = "🚨 **Vastus blokeeriti järelkontrolli poolt.**"
                        log_data["final_status"] = "BLOCKED"

        st.session_state.last_response = main_answer
        st.session_state.last_post_analysis = post_analysis
        st.session_state.last_status = "OK" if is_safe else "BLOCKED"

    except Exception as e:
        st.error(f"Kriitiline viga: {e}")
        log_data["error"] = str(e)
        st.session_state.last_status = "ERROR"
    
    finally:
        log_data["total_duration"] = round(time.time() - start_time_total, 2)
        if is_safe and log_data["final_status"] == "PENDING":
            log_data["final_status"] = "OK"
        
        log_json_event(log_data)
        st.session_state.processing = False
        st.rerun()

# --- VÄLJUNDI KUVAMINE ---
if st.session_state.last_response:
    st.divider()
    if st.session_state.last_status == "OK":
        st.subheader("💡 Tehisintellekti vastus:")
        st.markdown(st.session_state.last_response)
        if st.session_state.last_post_analysis:
            with st.expander("🛡️ Kvaliteedikontrolli selgitus", expanded=False):
                st.info(st.session_state.last_post_analysis)
    else:
        st.warning(st.session_state.last_response)
        if st.session_state.last_post_analysis:
            with st.expander("🛡️ Miks vastus blokeeriti?", expanded=True):
                st.error(st.session_state.last_post_analysis)

# --- LOGID UI-S ---
if st.session_state.show_logs:
    st.divider()
    st.subheader("📋 Viimased tegevused")
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in reversed(lines[-10:]):
                try:
                    entry = json.loads(line)
                    mumm = "🟢" if entry.get("final_status") == "OK" else "🔴"
                    label = f"{mumm} {entry.get('timestamp')} | {entry.get('user_input', '')[:40]}..."
                    with st.expander(label):
                        st.json(entry)
                except:
                    st.text(line)