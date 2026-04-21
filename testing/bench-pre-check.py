import time
import json
import os
import requests
from datetime import datetime
from zoneinfo import ZoneInfo
from requests.auth import HTTPBasicAuth

# --- KONFIGURATSIOON ---
# API aadress konteinerite vahelises võrgus (eeldame nime 'api')
API_URL = "http://api:8000/pre-check"
API_USER = "admin"
API_PASS = "parool"

# Mudelid, mida võrdleme
MODELS_TO_TEST = ["gemma2:2b", "mistral", "llama3:8b", "phi3"]

# Failide asukohad
DATASET_FILE = "/testing/pre_check_dataset.json"
LOG_FILE = "/testing/bench-pre-check-log.json"


def ee_now_str() -> str:
    return datetime.now(ZoneInfo("Europe/Tallinn")).strftime("%Y-%m-%d %H:%M:%S")

def load_existing_log():
    """Laeb olemasoleva logi; tagastab tühja listi vigade korral."""
    if not os.path.exists(LOG_FILE):
        return []
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        # Kui olemasolev logi on vigane JSON, ei katkesta benchmarki jooksu
        return []

def load_dataset():
    """Laeb testandmed JSON failist."""
    if not os.path.exists(DATASET_FILE):
        print(f"❌ VIGA: Testandmete faili {DATASET_FILE} ei leitud!")
        return []
    try:
        with open(DATASET_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ VIGA andmete laadimisel: {e}")
        return []

def validate_case(expected_status, must_contain, must_contain_any, must_not_contain, normalized):
    """Valideerib ALLOWED/BLOCKED tulemuse ja tagastab detailse kontrolli."""
    missing_all = [kw for kw in must_contain if kw.lower() not in normalized]
    missing_any_groups = []
    forbidden_found = [kw for kw in must_not_contain if kw.lower() in normalized]

    for group in must_contain_any:
        lowered_group = [kw.lower() for kw in group if isinstance(kw, str) and kw.strip()]
        if lowered_group and not any(kw in normalized for kw in lowered_group):
            missing_any_groups.append(group)

    checks_ok = not missing_all and not missing_any_groups and not forbidden_found
    is_valid = expected_status == "BLOCKED" or checks_ok

    fail_reason = None
    if not is_valid:
        if missing_all or missing_any_groups:
            fail_reason = "missing_keywords"
        elif forbidden_found:
            fail_reason = "forbidden_keywords_present"

    return is_valid, fail_reason, missing_all, missing_any_groups, forbidden_found

def run_benchmark():
    dataset = load_dataset()
    if not dataset:
        print("⚠️ Testandmed puuduvad. Lõpetan.")
        return

    print(f"🚀 ALUSTAN BENCHMARKINGUT ({len(dataset)} küsimust, {len(MODELS_TO_TEST)} mudelit)")
    
    run_timestamp = ee_now_str()
    full_log = []
    summary_results = []

    for model in MODELS_TO_TEST:
        print(f"\n📦 Mudel: {model}")
        model_stats = {
            "model": model,
            "passed": 0,
            "total": len(dataset),
            "latencies": []
        }

        for case in dataset:
            case_id = case.get("id", "UNKNOWN")
            user_input = case.get("question")
            meta = case.get("pre_check_metadata", {})
            desc = meta.get("desc", "Kirjeldus puudub")
            category = meta.get("category", "uncategorized")
            difficulty = meta.get("difficulty", "unknown")
            expected_status = str(meta.get("expected_status", "ALLOWED")).upper()
            must_contain = meta.get("must_contain", [])
            must_contain_any = meta.get("must_contain_any", [])
            must_not_contain = meta.get("must_not_contain", [])

            # Koostame päringu keha koos pikema timeoutiga API jaoks
            payload = {
                "user_input": user_input,
                "model": model,
                "timeout": 360
            }

            start_time = time.time()
            try:
                # Teostame API päringu (Pythoni requests timeout on 365, et API jõuaks enne vastata)
                response = requests.post(
                    API_URL, 
                    json=payload, 
                    auth=HTTPBasicAuth(API_USER, API_PASS),
                    timeout=365
                )
                
                latency = time.time() - start_time
                model_stats["latencies"].append(latency)

                if response.status_code == 200:
                    res_data = response.json()
                    status = res_data.get("status")
                    normalized = res_data.get("normalized", "").lower()
                    
                    # Valideerime tulemuse
                    is_valid = False
                    fail_reason = None
                    missing_keywords = []
                    missing_any_groups = []
                    forbidden_found = []
                    if status == expected_status:
                        if status == "ALLOWED":
                            (
                                is_valid,
                                fail_reason,
                                missing_keywords,
                                missing_any_groups,
                                forbidden_found,
                            ) = validate_case(
                                expected_status,
                                must_contain,
                                must_contain_any,
                                must_not_contain,
                                normalized,
                            )
                        else:
                            is_valid = True
                    else:
                        fail_reason = "status_mismatch"

                    # Kuvame tulemuse ekraanil
                    if is_valid:
                        model_stats["passed"] += 1
                        print(f"  ✅ {desc:.<45} [{latency:.2f}s]")
                    else:
                        print(f"  ❌ {desc:.<45} [FAIL] -> {status} ({fail_reason})")

                    # Lisame detailsesse logisse
                    full_log.append({
                        "run_timestamp": run_timestamp,
                        "timestamp": ee_now_str(),
                        "model": model,
                        "case_id": case_id,
                        "desc": desc,
                        "category": category,
                        "difficulty": difficulty,
                        "input": user_input,
                        "expected_status": expected_status,
                        "expected_must_contain": must_contain,
                        "expected_must_contain_any": must_contain_any,
                        "expected_must_not_contain": must_not_contain,
                        "api_response": res_data,
                        "latency_sec": round(latency, 3),
                        "fail_reason": fail_reason,
                        "missing_keywords": missing_keywords,
                        "missing_any_groups": missing_any_groups,
                        "forbidden_found": forbidden_found,
                        "success": is_valid
                    })
                else:
                    print(f"  ⚠️ API VIGA ({model}): {response.status_code}")
                    full_log.append({
                        "run_timestamp": run_timestamp,
                        "timestamp": ee_now_str(),
                        "model": model,
                        "input": user_input,
                        "error": f"HTTP {response.status_code}",
                        "fail_reason": "http_error",
                        "success": False
                    })

            except Exception as e:
                print(f"  🔥 SIDEVIGA ({model}): {e}")
                full_log.append({
                    "run_timestamp": run_timestamp,
                    "timestamp": ee_now_str(),
                    "model": model,
                    "case_id": case_id,
                    "desc": desc,
                    "category": category,
                    "difficulty": difficulty,
                    "input": user_input,
                    "expected_status": expected_status,
                    "fail_reason": "exception",
                    "error": str(e),
                    "success": False
                })
        
        summary_results.append(model_stats)

    # Kirjutame koondlogi faili
    try:
        existing_log = load_existing_log()
        combined_log = existing_log + full_log
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(combined_log, f, ensure_ascii=False, indent=2)
        print(
            f"\n💾 Detailne logi salvestatud: {LOG_FILE} "
            f"(lisati {len(full_log)} uut kirjet, kokku {len(combined_log)})"
        )
    except Exception as e:
        print(f"❌ VIGA logi salvestamisel: {e}")

    # LÕPPKOKKUVÕTE: Koondtabeli väljastamine ekraanile
    print("\n" + "="*85)
    print(f"{'MUDEL':<20} | {'EDUKUS %':<12} | {'KESKM. LATENTSUS (s)':<20}")
    print("-" * 85)
    for r in summary_results:
        avg_lat = sum(r["latencies"]) / len(r["latencies"]) if r["latencies"] else 0
        acc = (r["passed"] / r["total"]) * 100 if r["total"] > 0 else 0
        print(f"{r['model']:<20} | {acc:>10.1f}% | {avg_lat:>18.3f}s")
    print("="*85)

if __name__ == "__main__":
    # Ootame hetke, et API jõuaks startida, kui käivitatakse koos
    time.sleep(1)
    run_benchmark()