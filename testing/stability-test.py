import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import requests
from requests.auth import HTTPBasicAuth


API_BASE_URL = os.getenv("STABILITY_API_BASE_URL", "http://api:8000")
API_USER = os.getenv("STABILITY_API_USER", "admin")
API_PASS = os.getenv("STABILITY_API_PASS", "parool")
DATASET_FILE = os.getenv("STABILITY_DATASET_FILE", "/testing/retrieval_dataset.json")
LOG_FILE = os.getenv("STABILITY_LOG_FILE", "/testing/stability-test-log.json")
TESTS_CONF_FILE = os.getenv("TESTS_CONF_FILE", "/testing/tests_conf.json")

DEFAULT_REPEAT = int(os.getenv("STABILITY_REPEAT", "20"))
DEFAULT_TIMEOUT = int(os.getenv("STABILITY_TIMEOUT", "180"))
DEFAULT_MODEL = os.getenv("STABILITY_MAIN_MODEL", "llama3:8b")
DEFAULT_THREADS = int(os.getenv("STABILITY_THREADS", "4"))
DEFAULT_N_RESULTS = int(os.getenv("STABILITY_N_RESULTS", "5"))
DEFAULT_MAX_CONTEXT_BLOCKS = int(os.getenv("STABILITY_MAX_CONTEXT_BLOCKS", "3"))


def ee_now_datetime() -> datetime:
    return datetime.now(ZoneInfo("Europe/Tallinn"))


def ee_now_str() -> str:
    return ee_now_datetime().strftime("%Y-%m-%d %H:%M:%S")


def print_step(message: str) -> None:
    print(f"[{ee_now_datetime().strftime('%H:%M:%S')}] {message}", flush=True)


def normalize_text(value: str) -> str:
    text = (value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_dataset(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        print_step(f"VIGA: testandmete faili {path} ei leitud.")
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, list) else []
    except Exception as exc:
        print_step(f"VIGA andmete laadimisel: {exc}")
        return []


def load_test_config() -> Dict[str, Any]:
    config = {
        "question": "",
        "repeat": DEFAULT_REPEAT,
        "threads": DEFAULT_THREADS,
        "timeout": DEFAULT_TIMEOUT,
        "n_results": DEFAULT_N_RESULTS,
        "max_context_blocks": DEFAULT_MAX_CONTEXT_BLOCKS,
        "pause_seconds": 0,
    }

    try:
        with open(TESTS_CONF_FILE, "r", encoding="utf-8-sig") as handle:
            tests_conf = json.load(handle)

        server_max = os.cpu_count() or 1
        tests = tests_conf.get("tests", {}) if isinstance(tests_conf, dict) else {}
        stability_conf = tests.get("stability-test", {})

        if isinstance(stability_conf, dict):
            config["question"] = str(stability_conf.get("question", config["question"])).strip()
            config["repeat"] = int(stability_conf.get("repeat", config["repeat"]))
            config["threads"] = int(stability_conf.get("threads", config["threads"]))
            config["timeout"] = int(stability_conf.get("timeout", config["timeout"]))
            config["n_results"] = int(stability_conf.get("n_results", config["n_results"]))
            config["max_context_blocks"] = int(stability_conf.get("max_context_blocks", config["max_context_blocks"]))
            config["pause_seconds"] = int(stability_conf.get("pause_seconds", config["pause_seconds"]))
        elif stability_conf:
            config["threads"] = int(stability_conf)

        config["threads"] = max(1, min(config["threads"], server_max))
        config["repeat"] = max(2, config["repeat"])
        config["timeout"] = max(1, config["timeout"])
        config["n_results"] = max(1, min(config["n_results"], 25))
        config["max_context_blocks"] = max(1, min(config["max_context_blocks"], 25))
        config["pause_seconds"] = max(0, config["pause_seconds"])
    except Exception:
        pass

    return config


def load_existing_log_history(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if isinstance(existing, list):
            return existing
        if isinstance(existing, dict):
            return [existing]
    except Exception:
        return []
    return []


def save_log(path: str, log_data: Dict[str, Any]) -> None:
    try:
        history = load_existing_log_history(path)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(history + [log_data], handle, ensure_ascii=False, indent=2)
    except Exception as exc:
        print_step(f"VIGA logi salvestamisel: {exc}")


def post_json(endpoint: str, payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    response = requests.post(
        f"{API_BASE_URL}{endpoint}",
        json=payload,
        auth=HTTPBasicAuth(API_USER, API_PASS),
        timeout=timeout,
    )
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
    return response.json() if response.content else {}


def fetch_retrieval(
    query: str,
    n_results: int,
    max_context_blocks: int,
    timeout: int,
) -> Dict[str, Any]:
    return post_json(
        "/retrieval",
        {
            "query": query,
            "n_results": n_results,
            "max_context_blocks": max_context_blocks,
        },
        timeout,
    )


def fetch_main_answer(
    query: str,
    context: str,
    model: str,
    threads: int,
    timeout: int,
) -> Dict[str, Any]:
    return post_json(
        "/query",
        {
            "user_input": query,
            "context": context,
            "model": model,
            "threads": threads,
            "timeout": timeout,
        },
        timeout,
    )


def counter_summary(values: List[str]) -> Dict[str, Any]:
    counts = Counter(values)
    if not counts:
        return {
            "unique_count": 0,
            "most_common_hash": None,
            "most_common_count": 0,
            "stability_rate": 0.0,
        }
    most_common_hash, most_common_count = counts.most_common(1)[0]
    return {
        "unique_count": len(counts),
        "most_common_hash": most_common_hash,
        "most_common_count": most_common_count,
        "stability_rate": round(most_common_count / len(values), 4),
    }


def sample_by_hash(items: List[Dict[str, Any]], field: str, limit: int = 3) -> Dict[str, str]:
    samples: Dict[str, str] = {}
    for item in items:
        item_hash = item.get(f"{field}_normalized_hash")
        item_value = item.get(field, "")
        if item_hash and item_hash not in samples:
            samples[item_hash] = item_value[:800]
        if len(samples) >= limit:
            break
    return samples


def run_case(
    case: Dict[str, Any],
    case_index: int,
    repeat: int,
    main_model: str,
    threads: int,
    timeout: int,
    default_n_results: int,
    default_max_context_blocks: int,
    pause_seconds: int,
) -> Dict[str, Any]:
    query = case.get("question", "")
    n_results = int(case.get("stability_n_results", default_n_results))
    max_context_blocks = int(case.get("stability_max_context_blocks", default_max_context_blocks))

    print_step(f"[{case_index}] Alustan: {query} | kordusi={repeat}")

    iterations: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for iteration in range(1, repeat + 1):
        started = time.time()
        try:
            retrieval = fetch_retrieval(query, n_results, max_context_blocks, timeout)
            context = retrieval.get("context", "")
            answer_data = fetch_main_answer(query, context, main_model, threads, timeout)
            answer = answer_data.get("result", "")

            context_normalized = normalize_text(context)
            answer_normalized = normalize_text(answer)

            iterations.append({
                "iteration": iteration,
                "duration_sec": round(time.time() - started, 3),
                "retrieval_found": bool(retrieval.get("found")),
                "retrieval_duration_ms": retrieval.get("duration"),
                "query_duration_ms": answer_data.get("duration"),
                "context": context,
                "context_exact_hash": stable_hash(context),
                "context_normalized_hash": stable_hash(context_normalized),
                "answer": answer,
                "answer_exact_hash": stable_hash(answer),
                "answer_normalized_hash": stable_hash(answer_normalized),
            })
        except Exception as exc:
            errors.append({
                "iteration": iteration,
                "duration_sec": round(time.time() - started, 3),
                "error": str(exc),
            })

        if iteration == 1 or iteration == repeat or iteration % max(1, repeat // 10) == 0:
            print_step(f"[{case_index}] edenemine {iteration}/{repeat}")

        if pause_seconds > 0 and iteration < repeat:
            print_step(f"[{case_index}] paus {pause_seconds}s enne järgmist katset")
            time.sleep(pause_seconds)

    context_hashes = [item["context_normalized_hash"] for item in iterations]
    answer_hashes = [item["answer_normalized_hash"] for item in iterations]
    context_summary = counter_summary(context_hashes)
    answer_summary = counter_summary(answer_hashes)

    passed = (
        len(errors) == 0
        and context_summary["unique_count"] == 1
        and answer_summary["unique_count"] == 1
        and len(iterations) == repeat
    )

    print_step(
        f"[{case_index}] {'PASS' if passed else 'FAIL'} | "
        f"retrieval unique={context_summary['unique_count']} | "
        f"answer unique={answer_summary['unique_count']} | errors={len(errors)}"
    )

    return {
        "index": case_index,
        "timestamp": ee_now_str(),
        "question": query,
        "expected_section": case.get("expected_section"),
        "n_results": n_results,
        "max_context_blocks": max_context_blocks,
        "repeat": repeat,
        "main_model": main_model,
        "threads": threads,
        "timeout": timeout,
        "pause_seconds": pause_seconds,
        "passed": passed,
        "retrieval_stable": context_summary["unique_count"] == 1 and len(errors) == 0,
        "answer_stable": answer_summary["unique_count"] == 1 and len(errors) == 0,
        "retrieval_summary": context_summary,
        "answer_summary": answer_summary,
        "errors": errors,
        "samples": {
            "contexts_by_hash": sample_by_hash(iterations, "context"),
            "answers_by_hash": sample_by_hash(iterations, "answer"),
        },
        "iterations": iterations,
    }


def select_cases(
    dataset: List[Dict[str, Any]],
    question: Optional[str],
    case_index: Optional[int],
    limit: Optional[int],
) -> List[Dict[str, Any]]:
    if question:
        return [{
            "question": question,
        }]
    if case_index is not None:
        if case_index < 1 or case_index > len(dataset):
            raise ValueError(f"case-index peab olema vahemikus 1..{len(dataset)}")
        return [dataset[case_index - 1]]
    return dataset[:limit] if limit is not None else dataset


def build_summary(results: List[Dict[str, Any]], started_at: float) -> Dict[str, Any]:
    total = len(results)
    passed = sum(1 for result in results if result["passed"])
    retrieval_stable = sum(1 for result in results if result["retrieval_stable"])
    answer_stable = sum(1 for result in results if result["answer_stable"])
    return {
        "total_cases": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round((passed / total) * 100, 1) if total else 0.0,
        "retrieval_stable_cases": retrieval_stable,
        "answer_stable_cases": answer_stable,
        "total_runtime_sec": round(time.time() - started_at, 3),
    }


def parse_args() -> argparse.Namespace:
    test_config = load_test_config()
    parser = argparse.ArgumentParser(
        description="Kontrollib, kas sama paring annab sama retrieval'i ja sama pohivastuse."
    )
    parser.add_argument("--repeat", type=int, default=test_config["repeat"], help="Korduste arv iga testloo kohta.")
    parser.add_argument("--dataset", default=DATASET_FILE, help="Testandmete JSON fail.")
    parser.add_argument("--question", default=test_config["question"] or None, help="Uks konkreetne kusimus dataset'i asemel.")
    parser.add_argument("--case-index", type=int, default=None, help="Dataset'i 1-pohine testloo indeks.")
    parser.add_argument("--limit", type=int, default=None, help="Mitu dataset'i esimest testlugu kaivitada.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Pohiparingu mudel.")
    parser.add_argument("--threads", type=int, default=test_config["threads"], help="Mudeli loimede arv.")
    parser.add_argument("--timeout", type=int, default=test_config["timeout"], help="HTTP ja mudeli timeout sekundites.")
    parser.add_argument("--pause-seconds", type=int, default=test_config["pause_seconds"], help="Paus sekundites kahe korduse vahel.")
    parser.add_argument("--n-results", type=int, default=test_config["n_results"], help="Retrieval'i kandidaatide arv.")
    parser.add_argument(
        "--max-context-blocks",
        type=int,
        default=test_config["max_context_blocks"],
        help="Mitu kontekstiplokki retrieval tagastab.",
    )
    parser.add_argument("--log-file", default=LOG_FILE, help="Logifail.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repeat < 2:
        print_step("VIGA: --repeat peab olema vahemalt 2.")
        return 1

    dataset = load_dataset(args.dataset)
    try:
        cases = select_cases(dataset, args.question, args.case_index, args.limit)
    except ValueError as exc:
        print_step(f"VIGA: {exc}")
        return 1

    if not cases:
        print_step("Testandmed puuduvad. Lopetan.")
        return 1

    run_started = time.time()
    log_data: Dict[str, Any] = {
        "run_started_at": ee_now_str(),
        "config": {
            "api_base_url": API_BASE_URL,
            "dataset_file": args.dataset,
            "repeat": args.repeat,
            "main_model": args.model,
            "threads": args.threads,
            "timeout": args.timeout,
            "n_results": args.n_results,
            "max_context_blocks": args.max_context_blocks,
            "pause_seconds": args.pause_seconds,
            "case_count": len(cases),
        },
        "results": [],
        "summary": {},
    }

    print_step(f"ALUSTAN STABIILSUSE TESTI ({len(cases)} testlugu, kordusi={args.repeat})")
    print_step(
        f"API: {API_BASE_URL} | põhimudel: {args.model} | lõimi: {args.threads} | "
        f"n_results: {args.n_results} | max_context_blocks: {args.max_context_blocks} | "
        f"paus: {args.pause_seconds}s"
    )

    for index, case in enumerate(cases, start=1):
        result = run_case(
            case=case,
            case_index=index,
            repeat=args.repeat,
            main_model=args.model,
            threads=args.threads,
            timeout=args.timeout,
            default_n_results=args.n_results,
            default_max_context_blocks=args.max_context_blocks,
            pause_seconds=max(0, args.pause_seconds),
        )
        log_data["results"].append(result)
        log_data["summary"] = build_summary(log_data["results"], run_started)

    summary = build_summary(log_data["results"], run_started)
    log_data["summary"] = summary
    log_data["run_finished_at"] = ee_now_str()
    save_log(args.log_file, log_data)

    print("\n" + "=" * 72)
    print("STABIILSUSE TESTI KOKKUVOTE")
    print("-" * 72)
    print(f"Testilugusid kokku       : {summary['total_cases']}")
    print(f"PASS                     : {summary['passed']}")
    print(f"FAIL                     : {summary['failed']}")
    print(f"Pass rate                : {summary['pass_rate']:.1f}%")
    print(f"Retrieval stabiilne      : {summary['retrieval_stable_cases']}/{summary['total_cases']}")
    print(f"Pohivastus stabiilne     : {summary['answer_stable_cases']}/{summary['total_cases']}")
    print(f"Kogu jooksu aeg          : {summary['total_runtime_sec']:.3f}s")
    print(f"Logifail                 : {args.log_file}")
    print("=" * 72)

    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
