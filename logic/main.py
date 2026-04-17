import streamlit as st
import os
import time
import json
import logic_core

# --- KONFIGURATSIOON ---
# Muudetud asukoht vastavalt nõudele logida logic kataloogi
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
    
    # Laeme hetke promptid
    current_prompts = logic_core.PROMPTS
    
    # Kuvame sisu redigeeritava tekstina
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
                # Uuendame ka mälus olevat objekti
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
    user_input = st.text_input("Sisesta oma küsimus riigihangete kohta:", placeholder="nt: Millised on riigihanke piirmäärad?", disabled=st.session_state.processing)
    submit_button = st.form_submit_button("Saada päring")

if submit_button and user_input:
    st.session_state.processing = True
    st.session_state.current_query = user_input
    # Peidame promptide vaate kui päring algab
    st.session_state.edit_prompts = False 
    st.rerun()

# --- TÖÖTLUSLOOGIKA ---
if st.session_state.processing and st.session_state.current_query:
    u_input = st.session_state.current_query
    start_time = time.time()
    status_container = st.empty()
    
    # Logi ettevalmistus
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

    def update_ui(msg):
        elapsed = round(time.time() - start_time, 1)
        timer_placeholder.metric("Kestus", f"{elapsed} sek")
        status_container.info(msg)

    try:
        # 1. SAMM: EELKONTROLL (Vajadusel)
        if "Eelkontroll" in security_option or "Täiskontroll" in security_option:
            update_ui("🔍 Samm 1/4: Päringu valideerimine ja normaliseerimine...")
            pre_p = logic_core.PROMPTS.get("PRE_CHECK_PROMPT", "").format(u_input=u_input)
            
            step_start = time.time()
            pre_res = logic_core.ask_ollama(guard_model_input, pre_p, selected_threads, selected_timeout)
            status, normalized = logic_core.parse_pre_check(pre_res)
            
            log_data["steps"]["pre_check"] = {
                "status": status,
                "normalized": normalized,
                "duration": round(time.time() - step_start, 2),
                "raw_response": pre_res
            }

            if status == "ALLOWED":
                if normalized:
                    active_query = normalized
            else:
                is_safe = False
                main_answer = f"🚨 **Päring blokeeritud turvafiltri poolt.**\n\nSelgitus: {pre_res}"
                log_data["final_status"] = "BLOCKED_BY_PRECHECK"

        # 2. SAMM: RAG JA PÕHIPÄRING
        if is_safe:
            update_ui(f"📚 Samm 2/4: Otsin asjakohast infot: **{active_query}**")
            
            step_start = time.time()
            context = logic_core.get_context(active_query)
            log_data["steps"]["context_fetch"] = {
                "found": bool(context),
                "duration": round(time.time() - step_start, 2)
            }
            
            if not context:
                # Kui isegi normaliseeritud päringuga midagi ei leita
                is_safe = False
                main_answer = "Andmebaasist ei leitud küsimusele vastamiseks piisavalt infot."
                log_data["final_status"] = "NO_CONTEXT_FOUND"
            else:
                # 3. SAMM: PÕHIPÄRING MUDELILE
                update_ui(f"🧠 Samm 3/4: Genereerin vastust ({main_model_input})...")
                rag_p = logic_core.PROMPTS.get("RAG_PROMPT", "").format(context=context, query=active_query)
                
                step_start = time.time()
                main_answer = logic_core.ask_ollama(main_model_input, rag_p, selected_threads, selected_timeout)
                log_data["steps"]["main_generation"] = {
                    "duration": round(time.time() - step_start, 2)
                }
                
                # 4. SAMM: JÄRELKONTROLL (Vajadusel)
                if "järelkontroll" in security_option or "Täiskontroll" in security_option:
                    update_ui("🛡️ Samm 4/4: Teen vastuse kvaliteedikontrolli...")
                    post_p = logic_core.PROMPTS.get("POST_CHECK_PROMPT", "").format(
                        u_input=u_input, 
                        context=context, 
                        main_res=main_answer
                    )
                    
                    step_start = time.time()
                    post_res = logic_core.ask_ollama(guard_model_input, post_p, selected_threads, selected_timeout)
                    post_data = logic_core.parse_json_res(post_res)
                    
                    p_status = post_data.get("status", "BLOCKED")
                    post_analysis = post_data.get("reason", post_res)
                    
                    log_data["steps"]["post_check"] = {
                        "status": p_status,
                        "duration": round(time.time() - step_start, 2),
                        "analysis": post_analysis
                    }
                    
                    if p_status != "ALLOWED":
                        is_safe = False
                        main_answer = "🚨 **Vastus blokeeriti järelkontrolli poolt (hallutsinatsiooni või ebatäpsuse oht).**"
                        log_data["final_status"] = "BLOCKED_BY_POSTCHECK"

        # Tulemuste salvestamine sessiooni
        st.session_state.last_response = main_answer
        st.session_state.last_post_analysis = post_analysis
        st.session_state.last_status = "OK" if is_safe else log_data.get("final_status", "FAILED")

    except Exception as e:
        st.error(f"Kriitiline viga töötlusel: {e}")
        log_data["error"] = str(e)
        st.session_state.last_status = "ERROR"
    
    finally:
        log_data["total_duration"] = round(time.time() - start_time, 2)
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
            with st.expander("🛡️ Järelkontrolli analüüs ja audit", expanded=True):
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
            # Näitame 15 viimast sündmust
            for line in reversed(lines[-15:]):
                try:
                    entry = json.loads(line)
                    mumm = "🟢" if entry.get("final_status") == "OK" else "🔴"
                    timestamp = entry.get("timestamp", "---")
                    label = f"{mumm} {timestamp} | {entry.get('user_input', '')[:50]}..."
                    
                    with st.expander(label):
                        st.json(entry)
                except:
                    st.text(line)