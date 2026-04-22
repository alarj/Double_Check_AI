import json
import os
import re
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from requests.auth import HTTPBasicAuth

# --- KONFIGURATSIOON ---
API_URL = "http://api:8000/retrieval"
API_USER = "admin"
API_PASS = "parool"
DATASET_FILE = "/testing/retrieval_dataset.json"
LOG_FILE = "/testing/retr-test-log.json"
SECTION_MARKER = "\u00a7"
DEFAULT_N_RESULTS = 5
DEFAULT_MAX_CONTEXT_BLOCKS = 5


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
        print(f"VIGA: Faili {DATASET_FILE} ei leitud!")
        return []
    try:
        with open(DATASET_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"VIGA andmete laadimisel: {e}")
        return []


def fetch_context_via_api(query, n_results=DEFAULT_N_RESULTS, max_context_blocks=DEFAULT_MAX_CONTEXT_BLOCKS):
    payload = {
        "query": query,
        "n_results": n_results,
        "max_context_blocks": max_context_blocks,
    }
    response = requests.post(
        API_URL,
        json=payload,
        auth=HTTPBasicAuth(API_USER, API_PASS),
        timeout=30,
    )
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
    return response.json() if response.content else {}


def run_retrieval_benchmark():
    dataset = load_dataset()
    if not dataset:
        return

    # Filtreerime välja ainult need küsimused, millel on olemas expected_section.
    test_cases = [c for c in dataset if c.get("expected_section") is not None]

    print(f"ALUSTAN RETRIEVAL BENCHMARKINGUT ({len(test_cases)} asjakohast küsimust)")
    print("-" * 75)

    full_log = []

    top1_count = 0
    topK_count = 0
    sum_rank = 0
    latencies = []

    for case in test_cases:
        query = case.get("question")
        expected_sec = str(case.get("expected_section"))
        n_results = int(case.get("n_results", DEFAULT_N_RESULTS))
        max_context_blocks = int(case.get("max_context_blocks", DEFAULT_MAX_CONTEXT_BLOCKS))

        start_time = time.time()
        try:
            api_data = fetch_context_via_api(
                query,
                n_results=n_results,
                max_context_blocks=max_context_blocks,
            )
            context_str = api_data.get("context", "")

            latency = time.time() - start_time
            latencies.append(latency)

            # Jagame kogu tagastatud teksti plokkideks, et kontrollida leidude asetust.
            blocks = [b.strip() for b in context_str.split("--- ALLIKAS:") if b.strip()]

            rank = -1
            expected_pattern = rf"{re.escape(SECTION_MARKER)}\s*{re.escape(expected_sec)}(?!\d)"

            for i, block in enumerate(blocks):
                if re.search(expected_pattern, block):
                    rank = i + 1
                    break

            is_top1 = rank == 1
            is_topK = rank != -1

            if is_top1:
                top1_count += 1
            if is_topK:
                topK_count += 1
                sum_rank += rank

            if is_topK:
                print(f"  OK [{latency:5.2f}s] '{query[:40]:<40}' -> Leiti ({SECTION_MARKER}{expected_sec}) kohal {rank}")
            else:
                print(f"  FAIL [{latency:5.2f}s] '{query[:40]:<40}' -> EI LEITUD ({SECTION_MARKER}{expected_sec})")

            full_log.append({
                "timestamp": ee_now_str(),
                "question": query,
                "expected_section": expected_sec,
                "n_results": n_results,
                "max_context_blocks": max_context_blocks,
                "found": is_topK,
                "rank": rank if is_topK else None,
                "latency_sec": round(latency, 3),
                "api_response": api_data,
                "sources_returned": [b[:100] + "..." for b in blocks],
                "raw_context_preview": (context_str[:200] + "...") if context_str else "PUUDUB",
            })

        except Exception as e:
            print(f"  VIGA päringul '{query[:30]}': {e}")
            full_log.append({
                "timestamp": ee_now_str(),
                "question": query,
                "expected_section": expected_sec,
                "n_results": n_results,
                "max_context_blocks": max_context_blocks,
                "error": str(e),
            })

    try:
        existing_log = load_existing_log()
        combined_log = existing_log + full_log
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(combined_log, f, ensure_ascii=False, indent=2)
        print(
            f"\nDetailne logi salvestatud: {LOG_FILE} "
            f"(lisati {len(full_log)} uut kirjet, kokku {len(combined_log)})"
        )
    except Exception as e:
        print(f"VIGA logi salvestamisel: {e}")

    total = len(test_cases)
    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    avg_rank = sum_rank / topK_count if topK_count > 0 else 0

    print("\n" + "=" * 60)
    print("RETRIEVAL (RAG) TULEMUSED")
    print("-" * 60)
    print(f"Testitud küsimusi : {total}")
    print(f"Top-1 täpsus      : {(top1_count/total)*100:.1f}% ({top1_count}/{total})")
    print(f"Top-K täpsus      : {(topK_count/total)*100:.1f}% ({topK_count}/{total})")
    print(f"Keskmine positsioon: {avg_rank:.2f} (ainult leitud vastuste seas)")
    print(f"Keskmine latentsus: {avg_lat:.3f} sekundit")
    print("=" * 60)


if __name__ == "__main__":
    time.sleep(1)
    run_retrieval_benchmark()
