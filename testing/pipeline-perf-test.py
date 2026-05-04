import json
import os
import socket
import time
from datetime import datetime
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

import requests
from requests.auth import HTTPBasicAuth

TESTS_CONF_FILE = os.getenv("TESTS_CONF_FILE", "/testing/tests_conf.json")
TEST_NAME = "pipeline-perf-test"


def ee_now_str() -> str:
    return datetime.now(ZoneInfo("Europe/Tallinn")).strftime("%Y-%m-%d %H:%M:%S")


def die(msg: str) -> None:
    raise RuntimeError(msg)


def load_json_file(path: str) -> Any:
    if not os.path.exists(path):
        die(f"Faili ei leitud: {path}")
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_conf() -> Dict[str, Any]:
    conf = load_json_file(TESTS_CONF_FILE)
    cfg = (conf.get("tests", {}) or {}).get(TEST_NAME)
    if not isinstance(cfg, dict):
        die(f"Konfiplokk tests.{TEST_NAME} puudub või pole objekt.")
    required = [
        "api_base_url",
        "api_user",
        "api_pass",
        "dataset_file",
        "log_file",
        "threads",
        "timeout",
        "precheck_model",
        "normalizer_model",
        "main_model",
        "postcheck_quality_model",
        "postcheck_security_model",
        "n_results",
        "max_context_blocks",
        "request_timeout_margin_sec",
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        die(f"Konfis puuduvad väljad tests.{TEST_NAME}: {', '.join(missing)}")
    return cfg


def append_jsonl(path: str, entry: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def call_api(url: str, payload: Dict[str, Any], auth: HTTPBasicAuth, timeout: int) -> Dict[str, Any]:
    start = time.time()
    resp = requests.post(url, json=payload, auth=auth, timeout=timeout)
    latency = round(time.time() - start, 3)
    body = {}
    try:
        if resp.content:
            body = resp.json()
    except Exception:
        body = {}
    return {"status_code": resp.status_code, "latency_sec": latency, "body": body}


def run_case(
    case: Dict[str, Any],
    threads: int,
    run_label: str,
    base_url: str,
    auth: HTTPBasicAuth,
    timeout: int,
    req_timeout: int,
    precheck_model: str,
    normalizer_model: str,
    main_model: str,
    post_q_model: str,
    post_s_model: str,
    default_n_results: int,
    default_max_blocks: int,
) -> Dict[str, Any]:
    case_id = case.get("id", "pp-unknown")
    desc = str(case.get("desc", "")).strip()
    user_input = str(case.get("user_input", "")).strip()
    security_level = str(case.get("security_level", "full")).strip().lower()
    normalization_mode = str(case.get("normalization_mode", "external")).strip().lower()
    secret = bool(case.get("secret", False))
    allow_all_subjects = bool(case.get("allow_all_subjects", False))
    allow_personal_data = bool(case.get("allow_personal_data", False))
    allowed_subject_ids = case.get("allowed_subject_ids", []) or []
    allowed_tenant_ids = case.get("allowed_tenant_ids", []) or []
    n_results = int(case.get("n_results", default_n_results))
    max_blocks = int(case.get("max_context_blocks", default_max_blocks))

    t0 = time.time()
    steps: Dict[str, Dict[str, Any]] = {}
    status = "OK"
    err = ""

    active_query = user_input

    if security_level in {"full", "pre_only"}:
        pre_payload = {
            "user_input": user_input,
            "model": precheck_model,
            "normalization_mode": normalization_mode,
            "threads": threads,
            "timeout": timeout,
        }
        pre = call_api(f"{base_url}/pre-check", pre_payload, auth, req_timeout)
        steps["pre_check"] = pre
        if pre["status_code"] != 200:
            status = "ERROR"
            err = f"pre-check HTTP {pre['status_code']}"
        else:
            active_query = (pre["body"] or {}).get("normalized", user_input) or user_input
            if (pre["body"] or {}).get("status") == "BLOCKED":
                status = "BLOCKED"

    if status == "OK" and normalization_mode == "external":
        norm_payload = {
            "user_input": user_input,
            "model": normalizer_model,
            "threads": threads,
            "timeout": timeout,
        }
        norm = call_api(f"{base_url}/normalize", norm_payload, auth, req_timeout)
        steps["normalize"] = norm
        if norm["status_code"] == 200:
            active_query = (norm["body"] or {}).get("normalized", active_query) or active_query
        else:
            status = "ERROR"
            err = f"normalize HTTP {norm['status_code']}"

    context = ""
    sources_returned_raw: List[Dict[str, Any]] = []
    if status == "OK":
        ret_payload = {
            "query": active_query,
            "original_query": user_input,
            "n_results": n_results,
            "max_context_blocks": max_blocks,
            "secret": secret,
            "allowed_subject_ids": allowed_subject_ids,
            "allowed_tenant_ids": allowed_tenant_ids,
            "allow_all_subjects": allow_all_subjects,
            "allow_personal_data": allow_personal_data,
        }
        ret = call_api(f"{base_url}/retrieval", ret_payload, auth, req_timeout)
        steps["retrieval"] = ret
        if ret["status_code"] != 200:
            status = "ERROR"
            err = f"retrieval HTTP {ret['status_code']}"
        else:
            context = (ret["body"] or {}).get("context", "") or ""
            sources_returned_raw = (ret["body"] or {}).get("sources_returned_raw", []) or []

    ai_response = "Esitatud kontekstis info puudub."
    if status == "OK" and context.strip():
        q_payload = {
            "user_input": active_query,
            "context": context,
            "model": main_model,
            "threads": threads,
            "timeout": timeout,
        }
        qres = call_api(f"{base_url}/query", q_payload, auth, req_timeout)
        steps["main_query"] = qres
        if qres["status_code"] != 200:
            status = "ERROR"
            err = f"query HTTP {qres['status_code']}"
        else:
            ai_response = (qres["body"] or {}).get("result", ai_response) or ai_response

    if status == "OK" and security_level in {"full", "post_only"}:
        common_payload = {
            "ai_response": ai_response,
            "original_user_input": user_input,
            "normalized_query": active_query,
            "context": context,
            "threads": threads,
            "timeout": timeout,
            "secret": secret,
            "allowed_subject_ids": allowed_subject_ids,
            "allowed_tenant_ids": allowed_tenant_ids,
            "allow_all_subjects": allow_all_subjects,
            "allow_personal_data": allow_personal_data,
            "sources_returned_raw": sources_returned_raw,
        }
        q_payload = dict(common_payload)
        q_payload["model"] = post_q_model
        s_payload = dict(common_payload)
        s_payload["model"] = post_s_model

        p4a = call_api(f"{base_url}/post-check-quality", q_payload, auth, req_timeout)
        p4b = call_api(f"{base_url}/post-check-security", s_payload, auth, req_timeout)
        steps["post_check_quality"] = p4a
        steps["post_check_security"] = p4b

        q_status = str((p4a.get("body") or {}).get("status", "ALLOWED"))
        s_status = str((p4b.get("body") or {}).get("status", "ALLOWED"))
        if q_status == "BLOCKED" or s_status == "BLOCKED":
            status = "BLOCKED"

    total = round(time.time() - t0, 3)
    return {
        "case_id": case_id,
        "desc": desc,
        "status": status,
        "error": err,
        "total_duration_sec": total,
        "run_label": run_label,
        "threads_effective": threads,
        "inputs": {
            "user_input": user_input,
            "security_level": security_level,
            "normalization_mode": normalization_mode,
            "secret": secret,
            "allow_all_subjects": allow_all_subjects,
            "allow_personal_data": allow_personal_data,
            "allowed_subject_ids": allowed_subject_ids,
            "allowed_tenant_ids": allowed_tenant_ids,
            "n_results": n_results,
            "max_context_blocks": max_blocks,
        },
        "sizes": {
            "user_input_chars": len(user_input),
            "active_query_chars": len(active_query),
            "context_chars": len(context or ""),
            "sources_returned_raw_chars": len(json.dumps(sources_returned_raw, ensure_ascii=False)),
            "ai_response_chars": len(ai_response or ""),
        },
        "steps": steps,
    }


def run() -> None:
    cfg = load_conf()
    dataset = load_json_file(str(cfg["dataset_file"]))
    if not isinstance(dataset, list) or not dataset:
        die("Dataset peab olema mittetühi JSON list.")

    base_url = str(cfg["api_base_url"]).rstrip("/")
    auth = HTTPBasicAuth(str(cfg["api_user"]), str(cfg["api_pass"]))
    server_max = os.cpu_count() or 1
    threads_cfg = int(cfg["threads"])
    threads_cfg_effective = max(1, min(threads_cfg, server_max))
    timeout = int(cfg["timeout"])
    req_timeout = timeout + int(cfg["request_timeout_margin_sec"])

    precheck_model = str(cfg["precheck_model"])
    normalizer_model = str(cfg["normalizer_model"])
    main_model = str(cfg["main_model"])
    post_q_model = str(cfg["postcheck_quality_model"])
    post_s_model = str(cfg["postcheck_security_model"])
    default_n_results = int(cfg["n_results"])
    default_max_blocks = int(cfg["max_context_blocks"])
    log_file = str(cfg["log_file"])
    hostname = socket.gethostname()

    run_plans: List[Dict[str, Any]] = []
    if server_max > threads_cfg:
        run_plans.append({"label": "config_threads", "threads": threads_cfg_effective})
        run_plans.append({"label": "max_threads", "threads": server_max})
    else:
        run_plans.append({"label": "single_run", "threads": threads_cfg_effective})

    run_started_at = ee_now_str()
    print(f"ALUSTAN PIPELINE JÕUDLUSTESTI ({len(dataset)} juhtumit) | algus: {run_started_at}")
    print(f"threads(conf): {threads_cfg} | server_max: {server_max} | timeout: {timeout}s")
    print("Jooksuplaan: " + ", ".join([f"{p['label']}={p['threads']}" for p in run_plans]))

    for plan in run_plans:
        run_label = plan["label"]
        threads_effective = int(plan["threads"])
        print(f"\n--- RUN {run_label} | threads={threads_effective} ---")

        for i, case in enumerate(dataset, start=1):
            result = run_case(
                case=case,
                threads=threads_effective,
                run_label=run_label,
                base_url=base_url,
                auth=auth,
                timeout=timeout,
                req_timeout=req_timeout,
                precheck_model=precheck_model,
                normalizer_model=normalizer_model,
                main_model=main_model,
                post_q_model=post_q_model,
                post_s_model=post_s_model,
                default_n_results=default_n_results,
                default_max_blocks=default_max_blocks,
            )
            step_times = {k: v.get("latency_sec", 0) for k, v in result["steps"].items()}
            print(
                f"{i:02d}. {result['case_id']} | {result['desc']} | "
                f"run={run_label} | status={result['status']} | total={result['total_duration_sec']}s | steps={step_times}"
            )

            append_jsonl(
                log_file,
                {
                    "timestamp": ee_now_str(),
                    "test_name": TEST_NAME,
                    "server": {
                        "hostname": hostname,
                        "server_max_cores": server_max,
                        "threads_configured": threads_cfg,
                        "threads_effective": threads_effective,
                        "run_label": run_label,
                        "timeout_configured_sec": timeout,
                        "request_timeout_effective_sec": req_timeout,
                    },
                    "models": {
                        "precheck": precheck_model,
                        "normalizer": normalizer_model,
                        "main": main_model,
                        "post_quality": post_q_model,
                        "post_security": post_s_model,
                    },
                    **result,
                },
            )

    run_finished_at = ee_now_str()
    print("=".ljust(72, "="))
    print(f"LÕPP: {run_finished_at}")
    print(f"Logi: {log_file}")
    print("=".ljust(72, "="))


if __name__ == "__main__":
    run()
