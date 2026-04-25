import json
import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from requests.auth import HTTPBasicAuth

# --- KONFIGURATSIOON ---
API_URL = "http://api:8000/normalize"
API_USER = "admin"
API_PASS = "parool"

MODELS_TO_TEST = [
    "alarjoeste/estonian-normalizer",
    "gemma2:2b",
    "phi3",
    "llama3:8b",
]

# Mudelipõhised timeoutid: estonian-normalizer on aeglasem, seega anname rohkem aega.
MODEL_TIMEOUTS = {
    "alarjoeste/estonian-normalizer": 900,
    "gemma2:2b": 360,
    "phi3": 360,
    "llama3:8b": 360,
}

DATASET_FILE = "/testing/normalizer_dataset.json"
LOG_FILE = "/testing/normalizer-test-log.json"


def ee_now_str() -> str:
    return datetime.now(ZoneInfo("Europe/Tallinn")).strftime("%Y-%m-%d %H:%M:%S")


def load_existing_log():
    if not os.path.exists(LOG_FILE):
        return []
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def load_dataset():
    if not os.path.exists(DATASET_FILE):
        print(f"VIGA: Testandmete faili {DATASET_FILE} ei leitud!")
        return []
    try:
        with open(DATASET_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception as e:
        print(f"VIGA andmete laadimisel: {e}")
        return []


def validate_normalized(normalized_text: str, meta: dict):
    text = (normalized_text or "").lower().strip()
    must_contain = [str(x).lower() for x in meta.get("must_contain", []) if str(x).strip()]
    must_not_contain = [str(x).lower() for x in meta.get("must_not_contain", []) if str(x).strip()]
    must_contain_any = meta.get("must_contain_any", [])

    missing_all = [kw for kw in must_contain if kw not in text]
    forbidden_found = [kw for kw in must_not_contain if kw in text]
    missing_any_groups = []

    for group in must_contain_any:
        options = [str(x).lower() for x in group if str(x).strip()]
        if options and not any(opt in text for opt in options):
            missing_any_groups.append(group)

    is_valid = not missing_all and not missing_any_groups and not forbidden_found
    fail_reason = ""
    if not is_valid:
        if missing_all or missing_any_groups:
            fail_reason = "missing_keywords"
        elif forbidden_found:
            fail_reason = "forbidden_keywords_present"

    return is_valid, fail_reason, missing_all, missing_any_groups, forbidden_found


def run_benchmark():
    dataset = load_dataset()
    if not dataset:
        print("Testandmed puuduvad. Lõpetan.")
        return

    run_timestamp = ee_now_str()
    full_log = []
    summary = []

    print(f"ALUSTAN NORMALIZER TESTI ({len(dataset)} juhtumit, {len(MODELS_TO_TEST)} mudelit)")

    for model in MODELS_TO_TEST:
        print(f"\nMudel: {model}")
        passed = 0
        latencies = []

        for case in dataset:
            case_id = case.get("id", "UNKNOWN")
            user_input = case.get("question", "")
            meta = case.get("normalizer_metadata", {})
            desc = meta.get("desc", "Kirjeldus puudub")

            payload = {
                "user_input": user_input,
                "model": model,
                "timeout": MODEL_TIMEOUTS.get(model, 360),
                "threads": 4,
            }

            start_time = time.time()
            try:
                response = requests.post(
                    API_URL,
                    json=payload,
                    auth=HTTPBasicAuth(API_USER, API_PASS),
                    timeout=payload["timeout"] + 30,
                )
                latency = time.time() - start_time
                latencies.append(latency)

                if response.status_code == 200:
                    res_data = response.json()
                    normalized = res_data.get("normalized", "")
                    (
                        is_valid,
                        fail_reason,
                        missing_keywords,
                        missing_any_groups,
                        forbidden_found,
                    ) = validate_normalized(normalized, meta)

                    if is_valid:
                        passed += 1
                        print(f"  OK {desc:.<50} [{latency:.2f}s]")
                    else:
                        print(f"  FAIL {desc:.<48} ({fail_reason})")

                    full_log.append(
                        {
                            "run_timestamp": run_timestamp,
                            "timestamp": ee_now_str(),
                            "model": model,
                            "case_id": case_id,
                            "desc": desc,
                            "input": user_input,
                            "api_response": res_data,
                            "normalized": normalized,
                            "latency_sec": round(latency, 3),
                            "missing_keywords": missing_keywords,
                            "missing_any_groups": missing_any_groups,
                            "forbidden_found": forbidden_found,
                            "fail_reason": fail_reason if not is_valid else "",
                            "success": is_valid,
                        }
                    )
                else:
                    print(f"  API VIGA ({case_id}): {response.status_code}")
                    full_log.append(
                        {
                            "run_timestamp": run_timestamp,
                            "timestamp": ee_now_str(),
                            "model": model,
                            "case_id": case_id,
                            "input": user_input,
                            "error": f"HTTP {response.status_code}",
                            "fail_reason": "http_error",
                            "success": False,
                        }
                    )
            except Exception as e:
                latency = time.time() - start_time
                latencies.append(latency)
                print(f"  SIDEVIGA ({case_id}): {e}")
                full_log.append(
                    {
                        "run_timestamp": run_timestamp,
                        "timestamp": ee_now_str(),
                        "model": model,
                        "case_id": case_id,
                        "input": user_input,
                        "error": str(e),
                        "fail_reason": "exception",
                        "success": False,
                    }
                )

        total = len(dataset)
        avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
        acc = (passed / total) * 100 if total else 0.0
        summary.append(
            {
                "model": model,
                "passed": passed,
                "total": total,
                "accuracy": acc,
                "avg_latency_sec": avg_lat,
            }
        )

    try:
        existing_log = load_existing_log()
        combined_log = existing_log + full_log
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(combined_log, f, ensure_ascii=False, indent=2)
        print(
            f"\nDetailne logi salvestatud: {LOG_FILE} "
            f"(lisati {len(full_log)} kirjet, kokku {len(combined_log)})"
        )
    except Exception as e:
        print(f"VIGA logi salvestamisel: {e}")

    print("\n" + "=" * 82)
    print(f"{'MUDEL':<34} | {'EDUKUS %':<10} | {'KESKM. LATENTSUS (s)':<22}")
    print("-" * 82)
    for row in summary:
        print(
            f"{row['model']:<34} | "
            f"{row['accuracy']:>8.1f}% | "
            f"{row['avg_latency_sec']:>20.3f}s"
        )
    print("=" * 82)


if __name__ == "__main__":
    time.sleep(1)
    run_benchmark()
