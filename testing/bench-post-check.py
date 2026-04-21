import time
import json
import os
import requests
from datetime import datetime
from zoneinfo import ZoneInfo
from requests.auth import HTTPBasicAuth
from typing import Any, Dict, List, Optional

# --- KONFIGURATSIOON ---
API_URL = "http://api:8000/post-check"
API_USER = "admin"
API_PASS = "parool"

MODELS_TO_TEST = ["gemma2:2b", "mistral", "llama3:8b", "phi3"]

DATASET_FILE = "/testing/post_check_dataset.json"
LOG_FILE = "/testing/bench-post-check-log.json"


def ee_now_str() -> str:
    return datetime.now(ZoneInfo("Europe/Tallinn")).strftime("%Y-%m-%d %H:%M:%S")


def load_dataset() -> List[Dict[str, Any]]:
    if not os.path.exists(DATASET_FILE):
        print(f"❌ VIGA: Testandmete faili {DATASET_FILE} ei leitud!")
        return []
    try:
        with open(DATASET_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception as e:
        print(f"❌ VIGA andmete laadimisel: {e}")
        return []


def append_jsonl(path: str, entry: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def safe_upper(x: Optional[str], default: str) -> str:
    if not x:
        return default
    return str(x).strip().upper() or default


def run_benchmark() -> None:
    dataset = load_dataset()
    if not dataset:
        print("⚠️ Testandmed puuduvad. Lõpetan.")
        return

    run_timestamp = ee_now_str()
    print(f"🚀 ALUSTAN POST-CHECK BENCHMARKINGUT ({len(dataset)} juhtumit, {len(MODELS_TO_TEST)} mudelit)")

    summary_results: List[Dict[str, Any]] = []

    for model in MODELS_TO_TEST:
        print(f"\n📦 Mudel: {model}")

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
                "timeout": 360,
                "threads": 4,
            }

            start_time = time.time()
            api_status_code: Optional[int] = None
            res_data: Dict[str, Any] = {}
            error: Optional[str] = None

            try:
                response = requests.post(
                    API_URL,
                    json=payload,
                    auth=HTTPBasicAuth(API_USER, API_PASS),
                    timeout=365,
                )
                api_status_code = response.status_code
                latency = time.time() - start_time
                latencies.append(latency)

                if response.status_code == 200:
                    res_data = response.json() if response.content else {}
                    predicted_status = safe_upper(res_data.get("status"), "BLOCKED")

                    # Positiivne klass: "BLOCKED" (vigane/ebaturvaline vastus tuvastati)
                    if expected_status == "BLOCKED" and predicted_status == "BLOCKED":
                        tp += 1
                        print(f"  ✅ {case_id:.<20} [TP] [{latency:.2f}s]")
                    elif expected_status == "ALLOWED" and predicted_status == "BLOCKED":
                        fp += 1
                        print(f"  ❌ {case_id:.<20} [FP] [{latency:.2f}s]")
                    elif expected_status == "BLOCKED" and predicted_status == "ALLOWED":
                        fn += 1
                        print(f"  ❌ {case_id:.<20} [FN] [{latency:.2f}s]")
                    else:
                        tn += 1
                        print(f"  ✅ {case_id:.<20} [TN] [{latency:.2f}s]")
                else:
                    latency = time.time() - start_time
                    latencies.append(latency)
                    error = f"HTTP {response.status_code}"
                    print(f"  ⚠️ API VIGA ({case_id}): {response.status_code}")

            except Exception as e:
                latency = time.time() - start_time
                latencies.append(latency)
                error = str(e)
                print(f"  🔥 SIDEVIGA ({case_id}): {e}")

            log_entry = {
                "run_timestamp": run_timestamp,
                "timestamp": ee_now_str(),
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
            append_jsonl(LOG_FILE, log_entry)

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
    print(f"💾 Detailne logi (JSONL) lisatud faili: {LOG_FILE}")


if __name__ == "__main__":
    time.sleep(1)
    run_benchmark()

