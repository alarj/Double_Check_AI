import sys
import os
import json
import time
from datetime import datetime
import re

# --- KONTEINERI SEADISTUS ---
# Kuna jooksutame seda logic-app konteineris, on juurkataloog tavaliselt /app
# Kontrollime, kus asub logic_core
possible_paths = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), # arenduskeskkond
    "/app", # Docker konteiner
    os.getcwd()
]

logic_found = False
for p in possible_paths:
    if os.path.exists(os.path.join(p, 'logic_core.py')):
        sys.path.append(p)
        logic_found = True
        break

try:
    import logic_core
except ImportError:
    print("❌ Viga: logic_core.py ei leitud ühestki oodatud asukohast!")
    print(f"Otsitud radadelt: {possible_paths}")
    sys.exit(1)

# --- KONFIGURATSIOON ---
# Mudelid, mida testime
TEST_MODELS = ["gemma2:2b", "llama3:8b", "mistral", "phi3"]

# Konteinerisisesed asukohad (Docker-compose volüümid)
DATASET_PATH = "/app/data_pipeline/retrieval_dataset.json" 
# Kui fail on muus asukohas, proovime alternatiive
if not os.path.exists(DATASET_PATH):
    DATASET_PATH = "testing/retrieval_dataset.json"

RESULTS_DIR = "/app/storage/results" if os.path.exists("/app/storage") else "testing/results"

# Loome tulemuste kausta
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(RESULTS_DIR, f"pre_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

def run_benchmark():
    print(f"🚀 Alustan Pre-check benchmark testimist [{', '.join(TEST_MODELS)}]")
    print(f"📂 Andmestik: {DATASET_PATH}")
    
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Viga: Andmestikku ei leitud!")
        return

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    results = {
        "benchmark_type": "pre_check",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "environment": "logic-app-container",
        "models": {}
    }

    # Määrame lõimede arvu vastavalt konteineri ressurssidele (vaikimisi 4)
    threads = os.cpu_count() or 4

    for model_name in TEST_MODELS:
        print(f"\n🤖 Testin mudelit: {model_name}")
        model_results = []
        total_correct = 0
        total_duration = 0
        processed_count = 0

        for i, case in enumerate(test_cases):
            u_input = case.get("question", "")
            if not u_input: continue
            
            # Määrame oodatava staatuse: out_of_domain tähendab BLOCKED
            expected_type = case.get("type", "legal") 
            expected_status = "ALLOWED" if expected_type != "out_of_domain" else "BLOCKED"
            
            print(f"  [{i+1}/{len(test_cases)}] Küsimus: {u_input[:35]}...", end=" ", flush=True)
            
            # Võtame prompti logic_core-ist
            prompt_template = logic_core.PROMPTS.get("PRE_CHECK_PROMPT", "")
            if not prompt_template:
                print("❌ Viga: PRE_CHECK_PROMPT puudub!")
                break
                
            full_prompt = prompt_template.replace("{u_input}", u_input)
            
            start_time = time.time()
            try:
                # Päring Ollamasse (mis on tavaliselt samas võrgus nimega 'ollama')
                raw_res = logic_core.ask_ollama(model_name, full_prompt, threads=threads, timeout=45)
                
                # Parsime tulemuse logic_core-i robustse parsimisega
                parsed_data = logic_core.parse_json_res(raw_res)
                actual_status = parsed_data.get("status", "PARSING_ERROR")
                normalized = parsed_data.get("normalized_query", "")
                
                duration = time.time() - start_time
                is_correct = (actual_status == expected_status)
                
                if is_correct: total_correct += 1
                total_duration += duration
                processed_count += 1
                
                model_results.append({
                    "question": u_input,
                    "expected": expected_status,
                    "actual": actual_status,
                    "correct": is_correct,
                    "duration": round(duration, 3)
                })
                
                status_icon = "✅" if is_correct else "❌"
                print(f"{status_icon} ({actual_status}) {round(duration, 2)}s")
                
            except Exception as e:
                print(f"🔥 VIGA: {str(e)}")
                model_results.append({"question": u_input, "error": str(e)})

        results["models"][model_name] = {
            "accuracy": round(total_correct / processed_count, 3) if processed_count > 0 else 0,
            "avg_duration": round(total_duration / processed_count, 3) if processed_count > 0 else 0,
            "details": model_results
        }

    # Salvestame tulemused
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Benchmark lõpetatud! Tulemused: {RESULTS_FILE}")
    
    # Prindime kokkuvõtte
    print("\n📊 KONTEINERI PRE-CHECK VÕRDLUSTABEL:")
    print(f"{'Mudel':<15} | {'Täpsus (%)':<12} | {'Aeg (sek)':<10}")
    print("-" * 45)
    for m in TEST_MODELS:
        if m in results["models"]:
            d = results["models"][m]
            print(f"{m:<15} | {d['accuracy']*100:>10.1f}% | {d['avg_duration']:>10.3f}")

if __name__ == "__main__":
    run_benchmark()