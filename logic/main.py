import base64
import json
import os
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

import logic_core
import streamlit as st

# --- KONFIGURATSIOON ---
LOG_FILE = "/app/ai_turvakiht.log"
PROMPTS_FILE = "/app/prompts.json"
PROMPTS_CHANGE_LOG_FILE = "/app/prompts_change_log.json"
API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8000")
API_USER = os.getenv("API_USER", "admin")
API_PASSWORD = os.getenv("API_PASSWORD", "parool")
DEFAULT_GUARD = "gemma2:2b"
DEFAULT_MAIN = "llama3:8b"
DEFAULT_NORMALIZER = "alarjoeste/estonian-normalizer"
WORKSPACE_DIR = os.getenv("WORKSPACE_DIR", "/workspace")
total_cores = os.cpu_count() or 1


def log_json_event(data):
    """Kirjutab sündmuse logifaili JSON formaadis."""
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"VIGA LOGIMISEL: {e}")


def append_prompt_change_log(old_prompts, new_prompts):
    """Lisab promptide muudatuse auditikirje JSON-faili."""
    entry = {
        "timestamp": logic_core.get_ee_time().strftime("%Y-%m-%d %H:%M:%S"),
        "old_prompts": old_prompts,
        "new_prompts": new_prompts,
    }
    try:
        existing = []
        if os.path.exists(PROMPTS_CHANGE_LOG_FILE):
            with open(PROMPTS_CHANGE_LOG_FILE, "r", encoding="utf-8") as f:
                raw = f.read().strip()
                if raw:
                    parsed = json.loads(raw)
                    if isinstance(parsed, list):
                        existing = parsed
        existing.append(entry)
        os.makedirs(os.path.dirname(PROMPTS_CHANGE_LOG_FILE), exist_ok=True)
        with open(PROMPTS_CHANGE_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.error(f"Prompti muutuse logimine ebaõnnestus: {e}")


def fetch_logs_via_api(source, limit=50):
    """Toob logid REST API /logs endpointist."""
    try:
        auth_token = base64.b64encode(f"{API_USER}:{API_PASSWORD}".encode("utf-8")).decode("utf-8")
        query = urllib.parse.urlencode({"source": source, "limit": limit})
        url = f"{API_BASE_URL}/logs?{query}"
        req = urllib.request.Request(url, headers={"Authorization": f"Basic {auth_token}"})
        with urllib.request.urlopen(req, timeout=10) as res:
            data = json.loads(res.read().decode("utf-8"))
            return data.get("data", []), None
    except Exception as e:
        return [], str(e)


def fetch_retrieval_context_via_api(query_text, n_results=5, max_context_blocks=3, secret=False):
    """Toob RAG konteksti REST API /retrieval endpointist."""
    try:
        auth_token = base64.b64encode(f"{API_USER}:{API_PASSWORD}".encode("utf-8")).decode("utf-8")
        payload = json.dumps({
            "query": query_text,
            "n_results": n_results,
            "max_context_blocks": max_context_blocks,
            "secret": bool(secret),
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{API_BASE_URL}/retrieval",
            data=payload,
            headers={
                "Authorization": f"Basic {auth_token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as res:
            data = json.loads(res.read().decode("utf-8"))
            return data.get("context", ""), data, None
    except Exception as e:
        return "", None, str(e)


def fetch_precheck_via_api(user_input, model, normalization_mode, threads, timeout):
    """Toob eelkontrolli tulemuse REST API /pre-check endpointist."""
    try:
        auth_token = base64.b64encode(f"{API_USER}:{API_PASSWORD}".encode("utf-8")).decode("utf-8")
        payload = json.dumps({
            "user_input": user_input,
            "model": model,
            "normalization_mode": normalization_mode,
            "threads": int(threads),
            "timeout": int(timeout),
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{API_BASE_URL}/pre-check",
            data=payload,
            headers={
                "Authorization": f"Basic {auth_token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=int(timeout) + 30) as res:
            data = json.loads(res.read().decode("utf-8"))
            return data, None
    except Exception as e:
        return None, str(e)


def fetch_normalized_query_via_api(user_input, model, threads, timeout, gemini_api_key=""):
    """Toob normaliseeritud päringu REST API /normalize endpointist."""
    try:
        auth_token = base64.b64encode(f"{API_USER}:{API_PASSWORD}".encode("utf-8")).decode("utf-8")
        req_data = {
            "user_input": user_input,
            "model": model,
            "threads": int(threads),
            "timeout": int(timeout),
        }
        if gemini_api_key:
            req_data["gemini_api_key"] = gemini_api_key
        payload = json.dumps(req_data).encode("utf-8")
        req = urllib.request.Request(
            f"{API_BASE_URL}/normalize",
            data=payload,
            headers={
                "Authorization": f"Basic {auth_token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=int(timeout) + 30) as res:
            data = json.loads(res.read().decode("utf-8"))
            return data, None
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8")
            parsed = json.loads(body) if body else {}
            detail = parsed.get("detail") if isinstance(parsed, dict) else None
            if detail:
                return None, f"HTTP {e.code}: {detail}"
        except Exception:
            pass
        return None, f"HTTP {e.code}: {e.reason}"
    except Exception as e:
        return None, str(e)


def run_git_command(args):
    for cwd in (WORKSPACE_DIR, "/app"):
        if not os.path.exists(cwd):
            continue
        try:
            return subprocess.check_output(
                args,
                stderr=subprocess.DEVNULL,
                text=True,
                cwd=cwd,
            ).strip()
        except Exception:
            continue
    return ""


def get_git_dir(base_dir):
    git_path = os.path.join(base_dir, ".git")
    if os.path.isdir(git_path):
        return git_path
    if os.path.isfile(git_path):
        try:
            with open(git_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if content.startswith("gitdir:"):
                rel_path = content.split(":", 1)[1].strip()
                return os.path.normpath(os.path.join(base_dir, rel_path))
        except Exception:
            return ""
    return ""


def read_git_head_ref():
    for cwd in (WORKSPACE_DIR, "/app"):
        if not os.path.exists(cwd):
            continue
        git_dir = get_git_dir(cwd)
        if not git_dir:
            continue
        head_path = os.path.join(git_dir, "HEAD")
        try:
            with open(head_path, "r", encoding="utf-8") as f:
                head = f.read().strip()
            if head.startswith("ref:"):
                return cwd, git_dir, head.split(":", 1)[1].strip()
        except Exception:
            continue
    return "", "", ""


def detect_git_branch():
    """Tagastab git haru nime kui olemas."""
    branch = run_git_command(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    if branch:
        return branch

    _, _, head_ref = read_git_head_ref()
    if head_ref:
        return os.path.basename(head_ref)
    return "teadmata"


def detect_build_time():
    """Tagastab build aja või viimase commit'i aja."""
    build_time = os.getenv("BUILD_TIME", "").strip()
    if build_time and build_time != "teadmata":
        return build_time

    commit_time = run_git_command(
        ["git", "log", "-1", "--format=%cd", "--date=format:%Y-%m-%d %H:%M:%S"]
    )
    if commit_time:
        return commit_time

    _, git_dir, _ = read_git_head_ref()
    if git_dir:
        log_head_path = os.path.join(git_dir, "logs", "HEAD")
        try:
            if os.path.exists(log_head_path):
                timestamp = os.path.getmtime(log_head_path)
                return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
        except Exception:
            pass
    return "teadmata"


def get_sidebar_title():
    return "Turvaline AI"


def get_page_title():
    return "Sinu nutikas AI assistent"


def load_current_prompts():
    try:
        # BOM-safe lugemine, et prompts editor töötaks ka utf-8-sig failidega.
        with open(PROMPTS_FILE, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception:
        return logic_core.PROMPTS


def render_prompt_editor():
    if not st.session_state.edit_prompts:
        return

    with prompt_section:
        st.divider()
        st.subheader("\U0001F4AF Süsteemi promptide haldus")
        current_prompts = load_current_prompts()
        prompts_text = json.dumps(current_prompts, indent=2, ensure_ascii=False)
        st.json(current_prompts)

        with st.form("prompt_form"):
            new_prompts_json = st.text_area(
                "Muuda prompts.json sisu:",
                value=prompts_text,
                height=400,
            )
            col1, col2 = st.columns([1, 5])
            with col1:
                save_clicked = st.form_submit_button("Salvesta")
            with col2:
                cancel_clicked = st.form_submit_button("Sulge")

        if save_clicked:
            try:
                parsed_json = json.loads(new_prompts_json)
                old_prompts = current_prompts
                with open(PROMPTS_FILE, "w", encoding="utf-8") as f:
                    json.dump(parsed_json, f, indent=2, ensure_ascii=False)
                append_prompt_change_log(old_prompts, parsed_json)
                logic_core.PROMPTS = parsed_json
                st.success("Promptid salvestatud!")
                time.sleep(1)
                st.session_state.edit_prompts = False
                st.rerun()
            except Exception as e:
                st.error(f"Vigane JSON! Kontrolli süntaksit: {e}")

        if cancel_clicked:
            st.session_state.edit_prompts = False
            st.rerun()


def render_logs():
    if not st.session_state.show_logs:
        return

    with logs_section:
        st.divider()
        st.subheader("\U0001F4CB Viimased tegevused")
        log_options = [
            "ui",
            "api",
            "test-pre-check",
            "test-post-check",
            "test-llm",
            "test-retrieval",
            "test-stability",
            "test-benchmark-embeddings",
            "test-normalizer",
            "prompts-change",
        ]
        log_source = st.selectbox(
            "Logi allikas:",
            options=log_options,
            index=log_options.index(st.session_state.log_source) if st.session_state.log_source in log_options else 0,
        )
        st.session_state.log_source = log_source

        logs, err = fetch_logs_via_api(log_source, limit=50)
        if err:
            st.error(f"Logide päring API kaudu ebaõnnestus: {err}")
        elif not logs:
            st.info("Valitud allikas ei tagastanud logikirjeid.")
        else:
            for entry in reversed(logs):
                if not isinstance(entry, dict):
                    st.text(str(entry))
                    continue
                status_value = entry.get("final_status") or entry.get("status")
                if status_value in ("OK", "ALLOWED"):
                    marker = "\U0001F7E2"  # green circle
                elif status_value in ("BLOCKED", "ERROR"):
                    marker = "\U0001F534"  # red circle
                else:
                    marker = "\U0001F7E1"  # yellow circle
                title_text = entry.get("user_input") or entry.get("endpoint") or "logikirje"
                label = f"{marker} {entry.get('timestamp', '---')} | {str(title_text)[:40]}"
                with st.expander(label):
                    st.json(entry)


def render_status_messages():
    status_placeholder.empty()
    if st.session_state.status_messages:
        with status_placeholder.container():
            st.subheader("\U0001F50D Töötlemise sammud")
            for msg in st.session_state.status_messages:
                st.info(msg)


def render_response():
    if not st.session_state.last_response:
        response_placeholder.empty()
        return

    with response_placeholder.container():
        st.divider()
        if st.session_state.last_status == "OK":
            st.subheader("\U0001F4AC Tehisintellekti vastus:")
            st.markdown(st.session_state.last_response)
            if st.session_state.last_post_analysis:
                with st.expander("\U0001F6E1 Kvaliteedikontrolli selgitus", expanded=False):
                    st.info(st.session_state.last_post_analysis)
        else:
            st.warning(st.session_state.last_response)
            if st.session_state.last_post_analysis:
                with st.expander("\U0001F6E1 Miks vastus blokeeriti?", expanded=True):
                    st.error(st.session_state.last_post_analysis)


# --- UI SEADISTAMINE ---
st.set_page_config(page_title="Sinu nutikas AI assistent", layout="wide")

# Session state algväärtustamine
if "processing" not in st.session_state:
    st.session_state.processing = False
if "last_response" not in st.session_state:
    st.session_state.last_response = None
if "last_post_analysis" not in st.session_state:
    st.session_state.last_post_analysis = None
if "last_status" not in st.session_state:
    st.session_state.last_status = None
if "show_logs" not in st.session_state:
    st.session_state.show_logs = False
if "edit_prompts" not in st.session_state:
    st.session_state.edit_prompts = False
if "current_query" not in st.session_state:
    st.session_state.current_query = None
if "current_secret" not in st.session_state:
    st.session_state.current_secret = False
if "log_source" not in st.session_state:
    st.session_state.log_source = "ui"
if "status_messages" not in st.session_state:
    st.session_state.status_messages = []
if "last_elapsed_sec" not in st.session_state:
    st.session_state.last_elapsed_sec = 0.0

# --- SIDEBAR ---
with st.sidebar:
    st.title(f"\U0001F6E1 {get_sidebar_title()}")
    st.caption("*100% vibecoded*")
    is_disabled = st.session_state.processing
    build_time = detect_build_time()
    build_branch_env = os.getenv("BUILD_BRANCH", "").strip()
    build_branch = build_branch_env if build_branch_env and build_branch_env != "teadmata" else detect_git_branch()

    st.subheader("Serveri sätted")
    st.caption(f"Build aeg: {build_time}")
    st.caption(f"Git haru: {build_branch}")
    selected_threads = st.number_input(
        "Lõimi (threads):",
        1,
        total_cores,
        total_cores,
        disabled=is_disabled,
        help="Mitu CPU lõime antakse mudelile vastuse arvutamiseks. Kui serveris on vähem lõimi, kasutatakse serveri tegelikku ülempiiri.",
    )
    selected_timeout = st.number_input(
        "Timeout (sek):",
        30,
        1200,
        360,
        disabled=is_disabled,
        help="Maksimaalne aeg sekundites, kui kaua üks mudeli või API samm võib vastust oodata.",
    )
    selected_n_results = st.number_input(
        "Retrieval kandidaate:",
        1,
        25,
        5,
        disabled=is_disabled,
        help="Mitu tulemust retrieval päringule baasparameetrina küsitakse. Taustal võidakse ümberreastamiseks küsida rohkem kandidaate.",
    )
    selected_max_context_blocks = st.number_input(
        "Kontekstiplokke:",
        1,
        25,
        3,
        disabled=is_disabled,
        help="Mitu parimaks hinnatud kontekstiplokki antakse lõpuks põhimudelile vastuse koostamiseks.",
    )

    st.divider()
    st.subheader("Mudelid")
    pre_check_model_input = st.text_input("Pre-check mudel:", DEFAULT_GUARD, disabled=is_disabled)
    normalization_mode_label = st.selectbox(
        "Normaliseerimine:",
        options=[
            "Normaliseeri pre-check mudeliga",
            "Normaliseeri eraldi mudeliga",
            "Ära normaliseeri",
        ],
        index=0,
        disabled=is_disabled,
    )
    mode_map = {
        "Normaliseeri pre-check mudeliga": "precheck",
        "Normaliseeri eraldi mudeliga": "external",
        "Ära normaliseeri": "off",
    }
    normalization_mode = mode_map.get(normalization_mode_label, "precheck")
    normalizer_model_input = DEFAULT_NORMALIZER
    gemini_api_key_input = ""
    if normalization_mode == "external":
        _, normalize_cfg_col = st.columns([1, 12])
        with normalize_cfg_col:
            normalizer_choice = st.selectbox(
                "Normaliseerimise mudel:",
                options=[
                    "alarjoeste/estonian-normalizer",
                    "gemini:gemini-2.5-flash",
                    "gemma2:2b",
                    "llama3:8b",
                    "Custom...",
                ],
                index=0,
                disabled=is_disabled,
                key="normalizer_model_choice",
            )
            if normalizer_choice == "Custom...":
                normalizer_model_input = st.text_input(
                    "Custom normaliseerimise mudel:",
                    DEFAULT_NORMALIZER,
                    disabled=is_disabled,
                    key="normalizer_model_custom",
                )
            else:
                normalizer_model_input = normalizer_choice
            if normalizer_model_input.lower().startswith("gemini:"):
                gemini_api_key_input = st.text_input(
                    "Gemini API key:",
                    value="",
                    type="password",
                    disabled=is_disabled,
                    key="gemini_api_key_input",
                )
    main_model_input = st.text_input("Põhimudel (RAG):", DEFAULT_MAIN, disabled=is_disabled)
    post_check_model_input = st.text_input("Post-check mudel:", DEFAULT_GUARD, disabled=is_disabled)

    security_option = st.selectbox(
        "Turvalisuse tase:",
        options=[
            "Eelkontroll (küsimuse valideerimine) ja põhipäring",
            "Põhipäring ja järelkontroll (tulemuse valideerimine)",
            "Täiskontroll (Eelkontroll, põhipäring, järelkontroll)",
            "Ainult põhipäring",
        ],
        index=2,
        disabled=is_disabled,
    )

    st.divider()
    if st.button("\U0001F4CB Näita/Peida logid", disabled=is_disabled):
        st.session_state.show_logs = not st.session_state.show_logs
        st.rerun()

    if st.button("\U0001F4AF Muuda prompte", disabled=is_disabled):
        st.session_state.edit_prompts = not st.session_state.edit_prompts
        st.rerun()

    timer_placeholder = st.empty()

# --- PEALEHT ---
st.title("\U0001F680 Sinu nutikas AI assistent")
st.caption("Süsteem kasutab bge-m3 embeddinguid ja struktuurset riigihangete andmebaasi.")

# --- PÄRINGU VORM ---
with st.form(key="query_form", clear_on_submit=False):
    user_input = st.text_input(
        "Sisesta oma küsimus riigihangete kohta:",
        placeholder="nt: mis on riigihange?",
        disabled=st.session_state.processing,
        key="query_input",
    )
    secret_input = st.checkbox(
        "Luba salajane info",
        value=st.session_state.current_secret,
        disabled=st.session_state.processing,
        key="secret_input",
    )
    submit_button = st.form_submit_button("Saada päring", disabled=st.session_state.processing)

status_section = st.container()
response_section = st.container()
prompt_section = st.container()
logs_section = st.container()

with status_section:
    status_placeholder = st.empty()
with response_section:
    response_placeholder = st.empty()

if submit_button and user_input:
    st.session_state.processing = True
    st.session_state.current_query = user_input
    st.session_state.current_secret = bool(secret_input)
    st.session_state.status_messages = []
    st.session_state.last_elapsed_sec = 0.0
    st.session_state.last_response = None
    st.session_state.last_post_analysis = None
    st.session_state.last_status = None
    st.rerun()

# --- TÖÖTLUSLOOGIKA ---
if st.session_state.processing and st.session_state.current_query:
    u_input = st.session_state.current_query
    secret_allowed = bool(st.session_state.current_secret)
    start_time_total = time.time()
    # Tühjendame eelmise päringu nähtava väljundi kohe uue töötluse alguses.
    status_placeholder.empty()
    response_placeholder.empty()

    log_data = {
        "timestamp": logic_core.get_ee_time().strftime("%Y-%m-%d %H:%M:%S"),
        "user_input": u_input,
        "secret": secret_allowed,
        "security_level": security_option,
        "steps": {},
        "final_status": "PENDING",
        "total_duration": 0,
    }

    active_query = u_input
    is_safe = True
    main_answer = ""
    post_analysis = ""
    fetched_context = ""

    def update_ui(msg):
        elapsed = round(time.time() - start_time_total, 1)
        st.session_state.last_elapsed_sec = elapsed
        timer_placeholder.metric("Kestus", f"{elapsed} sek")
        st.session_state.status_messages.append(msg)
        render_status_messages()

    try:
        if any(x in security_option for x in ["Eelkontroll", "Täiskontroll"]):
            step_start = time.time()
            update_ui("\U0001F528 Samm 1/4: Päringu valideerimine...")
            pre_data, pre_error = fetch_precheck_via_api(
                u_input,
                pre_check_model_input,
                normalization_mode,
                selected_threads,
                selected_timeout,
            )
            if pre_error:
                raise RuntimeError(f"Pre-check API viga: {pre_error}")

            status = str(pre_data.get("status", "BLOCKED")).upper()
            normalized = pre_data.get("normalized", "") or u_input
            reason = pre_data.get("reason", "")
            pre_duration = round(time.time() - step_start, 2)
            update_ui(f"\u2705 Samm 1/4 valmis ({pre_duration} sek)")

            log_data["steps"]["pre_check"] = {
                "model": pre_check_model_input,
                "normalization_mode": normalization_mode,
                "status": status,
                "normalized": normalized,
                "reason": reason,
                "duration": round(pre_duration, 2),
                "api_response": pre_data or {},
            }

            if status != "ALLOWED":
                is_safe = False
                explain = reason or "Turvakontroll blokeeris päringu."
                main_answer = f"**Päring blokeeritud turvafiltri poolt.**\n\nSelgitus: {explain}"
                log_data["final_status"] = "BLOCKED"
            else:
                if normalization_mode == "precheck":
                    active_query = normalized
                elif normalization_mode == "external":
                    norm_start = time.time()
                    update_ui(f"\U0001F9E9 Samm 1b: Normaliseerin päringu eraldi mudeliga ({normalizer_model_input})...")
                    norm_data, norm_error = fetch_normalized_query_via_api(
                        u_input,
                        normalizer_model_input,
                        selected_threads,
                        selected_timeout,
                        gemini_api_key=gemini_api_key_input,
                    )
                    if norm_error:
                        raise RuntimeError(f"Normalize API viga: {norm_error}")
                    active_query = norm_data.get("normalized", "") or u_input
                    norm_duration = round(time.time() - norm_start, 2)
                    update_ui(f"\u2705 Samm 1b valmis ({norm_duration} sek)")
                    log_data["steps"]["normalize"] = {
                        "model": normalizer_model_input,
                        "normalized": active_query,
                        "duration": round(norm_duration, 2),
                        "api_response": norm_data or {},
                    }
                else:
                    active_query = u_input

        if is_safe:
            update_ui(f"\U0001F4DA Samm 2/4: Otsin konteksti | päring: {active_query}")
            ctx_start = time.time()
            fetched_context, retrieval_data, retrieval_error = fetch_retrieval_context_via_api(
                active_query,
                n_results=selected_n_results,
                max_context_blocks=selected_max_context_blocks,
                secret=secret_allowed,
            )
            if retrieval_error:
                raise RuntimeError(f"Retrieval API viga: {retrieval_error}")
            context_found = len(fetched_context.strip()) > 0
            ctx_duration = round(time.time() - ctx_start, 2)
            update_ui(f"\u2705 Samm 2/4 valmis ({ctx_duration} sek) | päring: {active_query}")

            log_data["steps"]["context_fetch"] = {
                "found": context_found,
                "secret": secret_allowed,
                "duration": ctx_duration,
                "api_response": retrieval_data or {},
            }

            if not context_found:
                is_safe = False
                main_answer = "Esitatud kontekstis info puudub."
                log_data["final_status"] = "NO_CONTEXT"
            else:
                update_ui(f"\U0001F9E0 Samm 3/4: Genereerin vastust ({main_model_input}) | päring: {active_query}")
                rag_p_template = logic_core.PROMPTS.get("RAG_PROMPT", "")
                rag_p = rag_p_template.replace("{context}", fetched_context).replace("{query}", active_query)

                step_start_main = time.time()
                main_answer = logic_core.ask_ollama(main_model_input, rag_p, selected_threads, selected_timeout)
                main_duration = round(time.time() - step_start_main, 2)
                update_ui(f"\u2705 Samm 3/4 valmis ({main_duration} sek) | päring: {active_query}")

                log_data["steps"]["main_query"] = {
                    "model": main_model_input,
                    "prompt": rag_p,
                    "start_time": logic_core.get_ee_time().strftime("%H:%M:%S"),
                    "context_used": fetched_context,
                    "result": main_answer,
                    "duration": main_duration,
                }

                if any(x in security_option for x in ["järelkontroll", "Täiskontroll"]):
                    update_ui("\U0001F6E1 Samm 4/4: Teen vastuse kvaliteedikontrolli...")
                    post_p_template = logic_core.PROMPTS.get("POST_CHECK_PROMPT", "")
                    post_p = (
                        post_p_template
                        .replace("{u_input}", u_input)
                        .replace("{context}", fetched_context)
                        .replace("{main_res}", main_answer)
                    )

                    step_start_post = time.time()
                    post_res = logic_core.ask_ollama(post_check_model_input, post_p, selected_threads, selected_timeout)
                    post_data = logic_core.parse_json_res(post_res)
                    p_status = post_data.get("status", "ALLOWED")
                    post_analysis = post_data.get("analysis", post_data.get("reason", post_res))
                    post_duration = round(time.time() - step_start_post, 2)
                    update_ui(f"\u2705 Samm 4/4 valmis ({post_duration} sek)")

                    log_data["steps"]["post_check"] = {
                        "model": post_check_model_input,
                        "prompt": post_p,
                        "start_time": logic_core.get_ee_time().strftime("%H:%M:%S"),
                        "status": p_status,
                        "duration": post_duration,
                        "analysis": post_analysis,
                        "raw_response": post_res,
                    }
                    if p_status == "BLOCKED":
                        is_safe = False
                        main_answer = "**Vastus blokeeriti järelkontrolli poolt.**"
                        log_data["final_status"] = "BLOCKED"

        st.session_state.last_response = main_answer
        st.session_state.last_post_analysis = post_analysis
        if log_data["final_status"] == "NO_CONTEXT":
            st.session_state.last_status = "NO_CONTEXT"
        else:
            st.session_state.last_status = "OK" if is_safe else "BLOCKED"

    except Exception as e:
        st.session_state.last_response = f"Kriitiline viga: {e}"
        st.session_state.last_post_analysis = None
        st.session_state.last_status = "ERROR"
        is_safe = False
        log_data["final_status"] = "ERROR"
        log_data["error"] = str(e)

    finally:
        log_data["total_duration"] = round(time.time() - start_time_total, 2)
        st.session_state.last_elapsed_sec = log_data["total_duration"]
        timer_placeholder.metric("Kestus", f"{round(st.session_state.last_elapsed_sec, 1)} sek")
        if "error" in log_data and log_data.get("final_status") == "PENDING":
            log_data["final_status"] = "ERROR"
        elif is_safe and log_data["final_status"] == "PENDING":
            log_data["final_status"] = "OK"

        log_json_event(log_data)
        st.session_state.processing = False
        st.rerun()

if st.session_state.last_elapsed_sec > 0:
    timer_placeholder.metric("Kestus", f"{round(st.session_state.last_elapsed_sec, 1)} sek")

render_status_messages()
render_response()
render_prompt_editor()
render_logs()

