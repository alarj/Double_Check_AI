import json
import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional

import requests
from requests.auth import HTTPBasicAuth

TESTS_CONF_FILE = os.getenv("TESTS_CONF_FILE", "/testing/tests_conf.json")
TEST_NAME = "bench-post-check"


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
        "api_url",
        "api_user",
        "api_pass",
        "dataset_file",
        "log_file",
        "threads",
        "timeout",
        "request_timeout_margin_sec",
        "models",
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        die(f"Konfis puuduvad väljad tests.{TEST_NAME}: {', '.join(missing)}")
    if not isinstance(cfg.get("models"), list) or not cfg["models"]:
        die("tests.bench-post-check.models peab olema mittetühi list.")
    return cfg


def load_dataset(path: str) -> List[Dict[str, Any]]:
    data = load_json_file(path)
    if not isinstance(data, list):
        die("Post-check dataset peab olema JSON list.")
    return data


def append_jsonl(path: str, entry: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def safe_upper(x: Optional[str], default: str) -> str:
    if not x:
        return default
    return str(x).strip().upper() or default


def run_benchmark() -> None:
    cfg = load_conf()
    dataset = load_dataset(str(cfg["dataset_file"]))
    if not dataset:
        print("Testandmed puuduvad. Lõpetan.")
        return

    api_url = str(cfg["api_url"])
    api_user = str(cfg["api_user"])
    api_pass = str(cfg["api_pass"])
    log_file = str(cfg["log_file"])
    model_list = [str(m) for m in cfg["models"]]
    timeout = int(cfg["timeout"])
    request_timeout_margin_sec = int(cfg["request_timeout_margin_sec"])
    threads_cfg = int(cfg["threads"])
    server_max = os.cpu_count() or 1
    threads = max(1, min(threads_cfg, server_max))

    run_timestamp = ee_now_str()
    print(f"ALUSTAN POST-CHECK BENCHMARKINGUT ({len(dataset)} juhtumit, {len(model_list)} mudelit) | algus: {run_timestamp}")
    print(f"threads: {threads} (konf: {threads_cfg}, server_max: {server_max}) | timeout: {timeout}s")

    summary_results: List[Dict[str, Any]] = []

    for model in model_list:
        print(f"\nMudel: {model}")
        tp = fp = fn = tn = 0
        latencies: List[float] = []

        for case in dataset:
            case_id = case.get("id", "UNKNOWN")
            original_user_input = case.get("original_user_input") or case.get("question") or ""
            normalized_query = case.get("normalized_query") or ""
            context = case.get("context") or ""
            ai_response = case.get("answer") or ""
            expected_status = safe_upper(case.get("expected_status"), "BLOCKED")

            payload = {
                "original_user_input": original_user_input,
                "normalized_query": normalized_query,
                "context": context,
                "ai_response": ai_response,
                "model": model,
                "timeout": timeout,
                "threads": threads,
            }

            start_time = time.time()
            api_status_code: Optional[int] = None
            res_data: Dict[str, Any] = {}
            error: Optional[str] = None

            try:
                response = requests.post(
                    api_url,
                    json=payload,
                    auth=HTTPBasicAuth(api_user, api_pass),
                    timeout=timeout + request_timeout_margin_sec,
                )
                api_status_code = response.status_code
                latency = time.time() - start_time
                latencies.append(latency)

                if response.status_code == 200:
                    res_data = response.json() if response.content else {}
                    predicted_status = safe_upper(res_data.get("status"), "BLOCKED")
                    if expected_status == "BLOCKED" and predicted_status == "BLOCKED":
                        tp += 1
                        print(f"  OK   {case_id:.<20} [TP] [{latency:.2f}s]")
                    elif expected_status == "ALLOWED" and predicted_status == "BLOCKED":
                        fp += 1
                        print(f"  FAIL {case_id:.<20} [FP] [{latency:.2f}s]")
                    elif expected_status == "BLOCKED" and predicted_status == "ALLOWED":
                        fn += 1
                        print(f"  FAIL {case_id:.<20} [FN] [{latency:.2f}s]")
                    else:
                        tn += 1
                        print(f"  OK   {case_id:.<20} [TN] [{latency:.2f}s]")
                else:
                    latency = time.time() - start_time
                    latencies.append(latency)
                    error = f"HTTP {response.status_code}"
                    print(f"  API VIGA ({case_id}): {response.status_code}")
            except Exception as e:
                latency = time.time() - start_time
                latencies.append(latency)
                error = str(e)
                print(f"  SIDEVIGA ({case_id}): {e}")

            log_entry = {
                "run_timestamp": run_timestamp,
                "timestamp": ee_now_str(),
                "test_name": TEST_NAME,
                "model": model,
                "case_id": case_id,
                "expected_status": expected_status,
                "input": {
                    "original_user_input": original_user_input,
                    "normalized_query": normalized_query,
                    "context": context,
                    "ai_response": ai_response,
                },
                "payload": payload,
                "api_status_code": api_status_code,
                "api_response": res_data,
                "latency_sec": round(latency, 3),
                "error": error,
            }
            append_jsonl(log_file, log_entry)

        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        total = tp + fp + fn + tn
        accuracy = ((tp + tn) / total) if total > 0 else 0.0
        avg_lat = (sum(latencies) / len(latencies)) if latencies else 0.0

        summary_results.append(
            {
                "model": model,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "accuracy": accuracy,
                "avg_latency_sec": avg_lat,
            }
        )

    run_finished_at = ee_now_str()
    print("\n" + "=" * 110)
    print(f"{'MUDEL':<18} | {'PRECISION':<9} | {'RECALL':<7} | {'ACCURACY':<8} | {'TP':>3} {'FP':>3} {'FN':>3} {'TN':>3} | {'AVG LAT (s)':>10}")
    print("-" * 110)
    for r in summary_results:
        print(
            f"{r['model']:<18} | "
            f"{r['precision']*100:>8.1f}% | "
            f"{r['recall']*100:>6.1f}% | "
            f"{r['accuracy']*100:>7.1f}% | "
            f"{r['tp']:>3} {r['fp']:>3} {r['fn']:>3} {r['tn']:>3} | "
            f"{r['avg_latency_sec']:>10.3f}"
        )
    print("=" * 110)
    print(f"Ajavahemik: {run_timestamp} -> {run_finished_at}")
    print(f"Detailne logi (JSONL) lisatud faili: {log_file}")


if __name__ == "__main__":
    run_benchmark()
