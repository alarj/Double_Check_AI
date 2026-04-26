import json
import os
import re
import time
import unicodedata
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


def normalize_text(text: str) -> str:
    """Normaliseerib teksti võrdluseks (diakriitika, eraldajad, tühikud)."""
    lowered = (text or "").lower().replace("_", " ").replace("-", " ")
    lowered = "".join(
        ch for ch in unicodedata.normalize("NFKD", lowered)
        if not unicodedata.combining(ch)
    )
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def build_candidates(normalized_text: str):
    """
    Koostab kontrollikandidaadid:
    - kogu tekst (substring kontrolliks),
    - tokenid,
    - 2-tokeni liited (nt 'lihtsa hanke' -> 'lihtsahanke').
    """
    tokens = re.findall(r"\w+", normalized_text)
    candidates = set(tokens)
    for i in range(len(tokens) - 1):
        candidates.add(tokens[i] + tokens[i + 1])
    return normalized_text, candidates


def trigram_similarity(a: str, b: str) -> float:
    """Lihtne trigrammi sarnasus morfoloogiliste variantide püüdmiseks."""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    def grams(s: str):
        s = f"  {s}  "
        return {s[i:i+3] for i in range(len(s) - 2)}

    ga = grams(a)
    gb = grams(b)
    if not ga or not gb:
        return 0.0
    inter = len(ga & gb)
    union = len(ga | gb)
    return inter / union if union else 0.0


def keyword_match(keyword: str, normalized_text: str, candidates: set) -> bool:
    """
    Kontrollib märksõna leidumist robustselt:
    1) otsene substring,
    2) täpne token/2-tokeni liide,
    3) trigrammi sarnasus (kõrge lävend), et taluda käändeid/ühendi varieerumist.
    """
    kw = normalize_text(keyword)
    if not kw:
        return True

    compact_kw = kw.replace(" ", "")
    compact_text = normalized_text.replace(" ", "")

    if kw in normalized_text or compact_kw in compact_text:
        return True
    if kw in candidates or compact_kw in candidates:
        return True

    # Morfoloogilise variatsiooni jaoks range fuzzy-match.
    # Lävend on piisavalt kõrge, et vältida juhuslikke vasteid.
    for cand in candidates:
        if len(cand) < 5 or len(compact_kw) < 5:
            continue
        # Kui algus on täiesti erinev, jätame vahele (vähendab valepositiive).
        if cand[:2] != compact_kw[:2]:
            continue
        if trigram_similarity(cand, compact_kw) >= 0.78:
            return True
    return False


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
    text = normalize_text(normalized_text)
    text_for_substring, candidates = build_candidates(text)
    must_contain = [normalize_text(str(x)) for x in meta.get("must_contain", []) if str(x).strip()]
    must_not_contain = [normalize_text(str(x)) for x in meta.get("must_not_contain", []) if str(x).strip()]
    must_contain_any = meta.get("must_contain_any", [])

    missing_all = [kw for kw in must_contain if not keyword_match(kw, text_for_substring, candidates)]
    forbidden_found = [kw for kw in must_not_contain if keyword_match(kw, text_for_substring, candidates)]
    missing_any_groups = []

    for group in must_contain_any:
        options = [normalize_text(str(x)) for x in group if str(x).strip()]
        if options and not any(keyword_match(opt, text_for_substring, candidates) for opt in options):
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
