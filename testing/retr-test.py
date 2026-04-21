import time
import json
import os
import sys
import re
from datetime import datetime
from zoneinfo import ZoneInfo

# --- KESKKONNA SEADISTAMINE ---
# Kuna skript jookseb test-app konteineris, lisame /app kausta otsinguteele,
# et Python suudaks importida logic_core.py faili.
sys.path.append('/app')

try:
    import logic_core
    print("✅ logic_core moodul edukalt laaditud.")
except ImportError as e:
    print(f"❌ VIGA: logic_core moodulit ei leitud! ({e})")
    sys.exit(1)

# --- KONFIGURATSIOON ---
DATASET_FILE = "/testing/retrieval_dataset.json"
LOG_FILE = "/testing/retr-test-log.json"


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
    """Laeb testandmed JSON failist."""
    if not os.path.exists(DATASET_FILE):
        print(f"❌ VIGA: Faili {DATASET_FILE} ei leitud!")
        return []
    try:
        with open(DATASET_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ VIGA andmete laadimisel: {e}")
        return []

def run_retrieval_benchmark():
    dataset = load_dataset()
    if not dataset:
        return

    # Filtreerime välja ainult need küsimused, millel on eksisteeriv 'expected_section'
    test_cases = [c for c in dataset if c.get("expected_section") is not None]

    print(f"🚀 ALUSTAN RETRIEVAL BENCHMARKINGUT ({len(test_cases)} asjakohast küsimust)")
    print("-" * 75)
    
    full_log = []
    
    # Mõõdikud
    top1_count = 0
    topK_count = 0
    sum_rank = 0
    latencies = []

    for case in test_cases:
        query = case.get("question")
        expected_sec = str(case.get("expected_section"))
        
        start_time = time.time()
        try:
            # Kutsume välja logic_core.py funktsiooni get_context
            context_str = logic_core.get_context(query, n_results=5)
            
            latency = time.time() - start_time
            latencies.append(latency)
            
            # Jagame kogu tagastatud teksti plokkideks (iga plokk algab '--- ALLIKAS:')
            # Nii saame otsida paragrahvi numbrit kogu plokist (nii pealkirjast kui sisust)
            blocks = [b.strip() for b in context_str.split('--- ALLIKAS:') if b.strip()]
            
            rank = -1
            # Kasutame regexi, et vältida osalisi vasteid (nt otsides § 14, ei taha me leida § 141)
            expected_pattern = r'§\s*' + re.escape(expected_sec) + r'(?!\d)'
            
            for i, block in enumerate(blocks):
                # Otsime paragrahvi kogu tekstiplokist
                if re.search(expected_pattern, block):
                    rank = i + 1
                    break
            
            is_top1 = (rank == 1)
            is_topK = (rank != -1)
            
            if is_top1:
                top1_count += 1
            if is_topK:
                topK_count += 1
                sum_rank += rank
                
            # Reaalajas ekraaniväljund
            if is_topK:
                print(f"  ✅ [{latency:5.2f}s] '{query[:40]:<40}' -> Leiti (§{expected_sec}) kohal {rank}")
            else:
                print(f"  ❌ [{latency:5.2f}s] '{query[:40]:<40}' -> EI LEITUD (§{expected_sec})")
            
            # Lisame andmed logisse
            full_log.append({
                "timestamp": ee_now_str(),
                "question": query,
                "expected_section": expected_sec,
                "found": is_topK,
                "rank": rank if is_topK else None,
                "latency_sec": round(latency, 3),
                "sources_returned": [b[:100] + "..." for b in blocks], # Salvestame esimesed 100 tähte igast plokist
                "raw_context_preview": (context_str[:200] + "...") if context_str else "PUUDUB"
            })
            
        except Exception as e:
            print(f"  🔥 VIGA päringul '{query[:30]}': {e}")
            full_log.append({
                "timestamp": ee_now_str(),
                "question": query,
                "expected_section": expected_sec,
                "error": str(e)
            })

    # Salvestame koondlogi faili
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

    # Arvutame lõplikud mõõdikud
    total = len(test_cases)
    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    avg_rank = sum_rank / topK_count if topK_count > 0 else 0
    
    # Lõppkokkuvõte ekraanile
    print("\n" + "="*60)
    print("📊 RETRIEVAL (RAG) TULEMUSED")
    print("-" * 60)
    print(f"Testitud küsimusi : {total}")
    print(f"Top-1 täpsus      : {(top1_count/total)*100:.1f}% ({top1_count}/{total})")
    print(f"Top-K täpsus      : {(topK_count/total)*100:.1f}% ({topK_count}/{total})")
    print(f"Keskmine positsioon: {avg_rank:.2f} (ainult leitud vastuste seas)")
    print(f"Keskmine latentsus: {avg_lat:.3f} sekundit")
    print("="*60)

if __name__ == "__main__":
    run_retrieval_benchmark()