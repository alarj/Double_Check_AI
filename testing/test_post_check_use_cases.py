import json
import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, Dict, List

import requests
from requests.auth import HTTPBasicAuth

TESTS_CONF_FILE = os.getenv("TESTS_CONF_FILE", "/testing/tests_conf.json")
TEST_NAME = "post-check-use-cases"


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
    tests = conf.get("tests", {})
    cfg = tests.get(TEST_NAME)
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
        "quality_model",
        "security_model",
        "run_combined_check",
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        die(f"Konfis puuduvad väljad tests.{TEST_NAME}: {', '.join(missing)}")
    return cfg


def append_jsonl(path: str, entry: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def norm_status(value: Any) -> str:
    return str(value or "").strip().upper()


def build_payload(case: Dict[str, Any], model: str, threads: int, timeout: int) -> Dict[str, Any]:
    inp = case.get("input", {}) or {}
    rights = case.get("rights", {}) or {}
    return {
        "ai_response": inp.get("ai_response", ""),
        "original_user_input": inp.get("original_user_input", ""),
        "normalized_query": inp.get("normalized_query", ""),
        "context": inp.get("context", ""),
        "model": model,
        "threads": int(threads),
        "timeout": int(timeout),
        "secret": bool(rights.get("secret", False)),
        "allowed_subject_ids": rights.get("allowed_subject_ids", []) or [],
        "allowed_tenant_ids": rights.get("allowed_tenant_ids", []) or [],
        "allow_all_subjects": bool(rights.get("allow_all_subjects", False)),
        "allow_personal_data": bool(rights.get("allow_personal_data", False)),
        "sources_returned_raw": case.get("sources_returned_raw", []) or [],
    }


def call_endpoint(url: str, payload: Dict[str, Any], auth: HTTPBasicAuth, timeout: int) -> Dict[str, Any]:
    start = time.time()
    response = requests.post(url, json=payload, auth=auth, timeout=timeout + 30)
    latency = round(time.time() - start, 3)
    body = {}
    try:
        if response.content:
            body = response.json()
    except Exception:
        body = {}
    return {
        "status_code": response.status_code,
        "latency_sec": latency,
        "body": body,
    }


def run() -> None:
    cfg = load_conf()
    dataset = load_json_file(cfg["dataset_file"])
    if not isinstance(dataset, list) or not dataset:
        die("Dataset peab olema mittetühi JSON list.")

    base_url = str(cfg["api_base_url"]).rstrip("/")
    auth = HTTPBasicAuth(str(cfg["api_user"]), str(cfg["api_pass"]))
    server_max = os.cpu_count() or 1
    threads_cfg = int(cfg["threads"])
    threads = max(1, min(threads_cfg, server_max))
    timeout = int(cfg["timeout"])
    quality_model = str(cfg["quality_model"])
    security_model = str(cfg["security_model"])
    run_combined = bool(cfg["run_combined_check"])
    log_file = str(cfg["log_file"])

    quality_url = f"{base_url}/post-check-quality"
    security_url = f"{base_url}/post-check-security"
    combined_url = f"{base_url}/post-check"

    run_started_at = ee_now_str()
    print(f"ALUSTAN POST-CHECK KASUTUSJUHTUDE TESTI ({len(dataset)} juhtumit) | algus: {run_started_at}")
    print(f"4a mudel: {quality_model} | 4b mudel: {security_model} | combined: {run_combined}")
    print(f"threads: {threads} (konf: {threads_cfg}, server_max: {server_max}) | timeout: {timeout}s")

    pass_count = 0
    total = 0

    for case in dataset:
        total += 1
        case_id = case.get("id", f"case-{total}")
        case_desc = str(case.get("desc", "")).strip()
        expect = case.get("expect", {}) or {}
        exp_quality = norm_status(expect.get("quality_status"))
        exp_security = norm_status(expect.get("security_status"))
        exp_combined = norm_status(expect.get("combined_status"))

        quality_payload = build_payload(case, quality_model, threads, timeout)
        security_payload = build_payload(case, security_model, threads, timeout)

        quality_res = call_endpoint(quality_url, quality_payload, auth, timeout)
        security_res = call_endpoint(security_url, security_payload, auth, timeout)

        q_status = norm_status((quality_res.get("body") or {}).get("status"))
        s_status = norm_status((security_res.get("body") or {}).get("status"))
        local_combined = "BLOCKED" if ("BLOCKED" in {q_status, s_status}) else "ALLOWED"

        combined_res = None
        c_status = ""
        if run_combined:
            combined_payload = build_payload(case, quality_model, threads, timeout)
            combined_payload["quality_model"] = quality_model
            combined_payload["security_model"] = security_model
            combined_res = call_endpoint(combined_url, combined_payload, auth, timeout)
            c_status = norm_status((combined_res.get("body") or {}).get("status"))

        checks_ok = True
        fail_reasons: List[str] = []

        if exp_quality and q_status != exp_quality:
            checks_ok = False
            fail_reasons.append(f"quality {q_status}!={exp_quality}")
        if exp_security and s_status != exp_security:
            checks_ok = False
            fail_reasons.append(f"security {s_status}!={exp_security}")
        if exp_combined and local_combined != exp_combined:
            checks_ok = False
            fail_reasons.append(f"local_combined {local_combined}!={exp_combined}")
        if run_combined and exp_combined and c_status != exp_combined:
            checks_ok = False
            fail_reasons.append(f"api_combined {c_status}!={exp_combined}")

        q_lat = quality_res.get("latency_sec")
        s_lat = security_res.get("latency_sec")
        c_lat = (combined_res or {}).get("latency_sec") if run_combined else None

        if checks_ok:
            pass_count += 1
            if run_combined:
                print(
                    f"OK   {case_id} | {case_desc} | q={q_status} ({q_lat}s) | s={s_status} ({s_lat}s) | c={c_status} ({c_lat}s)"
                )
            else:
                print(
                    f"OK   {case_id} | {case_desc} | q={q_status} ({q_lat}s) | s={s_status} ({s_lat}s)"
                )
        else:
            if run_combined:
                print(
                    f"FAIL {case_id} | {case_desc} | q={q_status} ({q_lat}s) | s={s_status} ({s_lat}s) | c={c_status} ({c_lat}s) -> {'; '.join(fail_reasons)}"
                )
            else:
                print(
                    f"FAIL {case_id} | {case_desc} | q={q_status} ({q_lat}s) | s={s_status} ({s_lat}s) -> {'; '.join(fail_reasons)}"
                )

        log_entry = {
            "timestamp": ee_now_str(),
            "test_name": TEST_NAME,
            "case_id": case_id,
            "desc": case.get("desc", ""),
            "expected": {
                "quality_status": exp_quality,
                "security_status": exp_security,
                "combined_status": exp_combined,
            },
            "actual": {
                "quality_status": q_status,
                "security_status": s_status,
                "combined_status_local": local_combined,
                "combined_status_api": c_status if run_combined else None,
            },
            "models": {
                "quality_model": quality_model,
                "security_model": security_model,
            },
            "latency_sec": {
                "quality": quality_res.get("latency_sec"),
                "security": security_res.get("latency_sec"),
                "combined": (combined_res or {}).get("latency_sec") if run_combined else None,
            },
            "http": {
                "quality_status_code": quality_res.get("status_code"),
                "security_status_code": security_res.get("status_code"),
                "combined_status_code": (combined_res or {}).get("status_code") if run_combined else None,
            },
            "payload": {
                "quality": quality_payload,
                "security": security_payload,
            },
            "responses": {
                "quality": quality_res.get("body"),
                "security": security_res.get("body"),
                "combined": (combined_res or {}).get("body") if run_combined else None,
            },
            "success": checks_ok,
            "fail_reasons": fail_reasons,
        }
        append_jsonl(log_file, log_entry)

    run_finished_at = ee_now_str()
    success_pct = round((pass_count / total) * 100, 1) if total else 0.0
    print("=".ljust(72, "="))
    print(f"TULEMUS: {pass_count}/{total} ({success_pct}%)")
    print(f"Ajavahemik: {run_started_at} -> {run_finished_at}")
    print(f"Logi: {log_file}")
    print("=".ljust(72, "="))


if __name__ == "__main__":
    run()
