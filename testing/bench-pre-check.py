import time
import json
import os
import requests
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

def run_benchmark():
    dataset = load_dataset()
    if not dataset:
        print("⚠️ Testandmed puuduvad. Lõpetan.")
        return

    print(f"🚀 ALUSTAN BENCHMARKINGUT ({len(dataset)} küsimust, {len(MODELS_TO_TEST)} mudelit)")
    
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
            user_input = case.get("question")
            meta = case.get("pre_check_metadata", {})
            desc = meta.get("desc", "Kirjeldus puudub")
            expected_status = meta.get("expected_status", "ALLOWED")
            must_contain = meta.get("must_contain", [])

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
                    if status == expected_status:
                        if status == "ALLOWED" and must_contain:
                            # Kontrollime kohustuslikke märksõnu
                            missing = [kw for kw in must_contain if kw.lower() not in normalized]
                            is_valid = len(missing) == 0
                        else:
                            is_valid = True

                    # Kuvame tulemuse ekraanil
                    if is_valid:
                        model_stats["passed"] += 1
                        print(f"  ✅ {desc:.<45} [{latency:.2f}s]")
                    else:
                        print(f"  ❌ {desc:.<45} [FAIL] -> {status}")

                    # Lisame detailsesse logisse
                    full_log.append({
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "model": model,
                        "desc": desc,
                        "input": user_input,
                        "api_response": res_data,
                        "latency_sec": round(latency, 3),
                        "success": is_valid
                    })
                else:
                    print(f"  ⚠️ API VIGA ({model}): {response.status_code}")
                    full_log.append({
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "model": model,
                        "input": user_input,
                        "error": f"HTTP {response.status_code}",
                        "success": False
                    })

            except Exception as e:
                print(f"  🔥 SIDEVIGA ({model}): {e}")
        
        summary_results.append(model_stats)

    # Kirjutame koondlogi faili
    try:
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(full_log, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Detailne logi salvestatud: {LOG_FILE}")
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