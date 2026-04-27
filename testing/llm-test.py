import json
import os
import re
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, Dict, List


# --- KESKKONNA SEADISTAMINE ---
# Test jookseb konteineris käsuga:
# docker exec -it test-app python /testing/llm-test.py
# Seetõttu lisame nii /app kui /app/logic otsinguteele.
sys.path.append("/app")
sys.path.append("/app/logic")

try:
    import logic_core
except ImportError as exc:
    print(f"VIGA: logic_core moodulit ei leitud ({exc})")
    sys.exit(1)


# --- KONFIGURATSIOON ---
DATASET_FILE = "/testing/main_llm_dataset.json"
LOG_FILE = "/testing/llm-test-log.json"

DEFAULT_MAIN_MODEL = os.getenv("LLM_TEST_MAIN_MODEL", "llama3:8b")
DEFAULT_JUDGE_MODEL = os.getenv("LLM_TEST_JUDGE_MODEL", "gemma2:2b")
DEFAULT_THREADS = int(os.getenv("LLM_TEST_THREADS", "4"))
DEFAULT_TIMEOUT = int(os.getenv("LLM_TEST_TIMEOUT", "300"))
DEFAULT_JUDGE_TIMEOUT = int(os.getenv("LLM_TEST_JUDGE_TIMEOUT", "300"))
KEYWORD_PASS_THRESHOLD = float(os.getenv("LLM_TEST_KEYWORD_THRESHOLD", "0.5"))
NO_CONTEXT_RESPONSE = "Esitatud kontekstis info puudub."


def print_step(message: str) -> None:
    print(f"[{ee_now_datetime().strftime('%H:%M:%S')}] {message}", flush=True)


def ee_now_datetime() -> datetime:
    return datetime.now(ZoneInfo("Europe/Tallinn"))


def ee_now_str() -> str:
    return ee_now_datetime().strftime("%Y-%m-%d %H:%M:%S")


def load_dataset() -> List[Dict[str, Any]]:
    if not os.path.exists(DATASET_FILE):
        print_step(f"VIGA: testandmete faili {DATASET_FILE} ei leitud.")
        return []
    try:
        with open(DATASET_FILE, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        print_step(f"VIGA andmete laadimisel: {exc}")
        return []


def save_log(log_data: Dict[str, Any]) -> None:
    try:
        with open(LOG_FILE, "w", encoding="utf-8") as handle:
            json.dump(log_data, handle, ensure_ascii=False, indent=2)
    except Exception as exc:
        print_step(f"VIGA logi salvestamisel: {exc}")


def load_existing_log_history() -> List[Dict[str, Any]]:
    if not os.path.exists(LOG_FILE):
        return []
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as handle:
            existing = json.load(handle)
            if isinstance(existing, list):
                return existing
            if isinstance(existing, dict):
                return [existing]
    except Exception:
        return []
    return []


def normalize_text(value: str) -> str:
    text = (value or "").lower().strip()
    text = re.sub(r"[`\"']", "", text)
    text = re.sub(r"^[\W_]+|[\W_]+$", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def contains_keyword(text: str, keyword: str) -> bool:
    return normalize_text(keyword) in normalize_text(text)


def keyword_label(keyword_group: Any) -> str:
    if isinstance(keyword_group, list):
        return " | ".join(str(item) for item in keyword_group)
    return str(keyword_group)


def count_expected_keywords(answer: str, expected_keywords: List[Any]) -> Dict[str, Any]:
    found = []
    missing = []

    for keyword_group in expected_keywords:
        variants = keyword_group if isinstance(keyword_group, list) else [keyword_group]
        matched_variant = next(
            (str(variant) for variant in variants if contains_keyword(answer, str(variant))),
            None,
        )

        if matched_variant is not None:
            found.append({
                "group": keyword_label(keyword_group),
                "matched": matched_variant,
            })
        else:
            missing.append(keyword_label(keyword_group))

    total = len(expected_keywords)
    score = 1.0 if total == 0 else len(found) / total
    return {
        "found": found,
        "missing": missing,
        "score": round(score, 3),
        "threshold": KEYWORD_PASS_THRESHOLD,
    }


def count_forbidden_keywords(answer: str, forbidden_keywords: List[str]) -> Dict[str, Any]:
    violations = [kw for kw in forbidden_keywords if contains_keyword(answer, kw)]
    score = 0.0 if violations else 1.0
    return {
        "violations": violations,
        "score": score,
    }


def build_main_prompt(question: str, context: str) -> str:
    rag_template = logic_core.PROMPTS.get("RAG_PROMPT", "{context}\n{query}")
    return rag_template.replace("{context}", context).replace("{query}", question)


def sanitize_judge_output(raw_response: str) -> str:
    text = (raw_response or "").strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_judge_response(raw_response: str) -> Dict[str, Any]:
    cleaned = sanitize_judge_output(raw_response)
    parsed = logic_core.parse_json_res(cleaned)
    if parsed:
        return {
            "parsed": parsed,
            "parse_error": False,
            "cleaned_response": cleaned,
        }

    decoder = json.JSONDecoder()
    candidates = [cleaned, cleaned.replace("\n", " ")]

    for candidate in candidates:
        try:
            obj, _ = decoder.raw_decode(candidate)
            if isinstance(obj, dict):
                return {
                    "parsed": obj,
                    "parse_error": False,
                    "cleaned_response": cleaned,
                }
        except Exception:
            continue

    start_index = cleaned.find("{")
    while start_index != -1:
        try:
            obj, _ = decoder.raw_decode(cleaned[start_index:])
            if isinstance(obj, dict):
                return {
                    "parsed": obj,
                    "parse_error": False,
                    "cleaned_response": cleaned,
                }
        except Exception:
            start_index = cleaned.find("{", start_index + 1)
            continue
        break

    return {
        "parsed": {},
        "parse_error": True,
        "cleaned_response": cleaned,
    }


def is_no_context_case(case: Dict[str, Any], context: str) -> bool:
    return case.get("type") in {"out_of_domain", "no_answer_in_context"} or not context.strip()


def is_expected_no_context_answer(answer: str, expected_answer: str = NO_CONTEXT_RESPONSE) -> bool:
    return normalize_text(answer) == normalize_text(expected_answer)


def generate_answer(question: str, context: str, model: str, threads: int, timeout: int) -> Dict[str, Any]:
    start_time = time.time()
    if not context.strip():
        return {
            "answer": NO_CONTEXT_RESPONSE,
            "prompt": None,
            "duration_sec": round(time.time() - start_time, 3),
            "used_model_call": False,
        }

    prompt = build_main_prompt(question, context)
    answer = logic_core.ask_ollama(model, prompt, threads, timeout)
    return {
        "answer": answer,
        "prompt": prompt,
        "duration_sec": round(time.time() - start_time, 3),
        "used_model_call": True,
    }


def judge_answer(
    question: str,
    context: str,
    answer: str,
    expected_keywords: List[Any],
    forbidden_keywords: List[str],
    model: str,
    threads: int,
    timeout: int,
) -> Dict[str, Any]:
    judge_prompt = f"""
Sa oled range LLM-vastuste hindaja.
Hinda vastust ainult antud konteksti põhjal.

Kui kontekst puudub või kontekst ei sisalda küsimusele vastust, ja vastus
ütleb sisuliselt "Esitatud kontekstis info puudub",
siis see EI OLE hallutsinatsioon ja verdict peab olema PASS.

Hallutsinatsioon tähendab ainult seda, et vastuses lisatakse infot,
mida antud kontekst ei toeta.

KÜSIMUS:
{question}

KONTEKST:
{context if context else "PUUDUB"}

VASTUS:
{answer}

OODATUD MÄRKSÕNAD:
{json.dumps(expected_keywords, ensure_ascii=False)}

KEELATUD MÄRKSÕNAD:
{json.dumps(forbidden_keywords, ensure_ascii=False)}

Tagasta ainult JSON kujul:
{{
  "relevance_score": 0-5,
  "groundedness_score": 0-5,
  "terminology_score": 0-5,
  "hallucination": true/false,
  "verdict": "PASS" või "FAIL",
  "analysis": "lühike põhjendus eesti keeles"
}}
""".strip()

    start_time = time.time()
    raw_response = logic_core.ask_ollama(model, judge_prompt, threads, timeout)
    parse_result = parse_judge_response(raw_response)

    return {
        "model": model,
        "duration_sec": round(time.time() - start_time, 3),
        "prompt": judge_prompt,
        "raw_response": raw_response,
        "cleaned_response": parse_result["cleaned_response"],
        "parsed": parse_result["parsed"],
        "parse_error": parse_result["parse_error"],
    }


def evaluate_case(case: Dict[str, Any], generated_answer: str, judge_data: Dict[str, Any]) -> Dict[str, Any]:
    context = case.get("context", "")
    expected_keywords = case.get("expected_keywords", [])
    forbidden_keywords = case.get("forbidden_keywords", [])
    expected_answer = case.get("expected_answer")

    expected_eval = count_expected_keywords(generated_answer, expected_keywords)
    forbidden_eval = count_forbidden_keywords(generated_answer, forbidden_keywords)

    exact_match = None
    if expected_answer is not None:
        exact_match = normalize_text(generated_answer) == normalize_text(expected_answer)

    judge_parsed = judge_data.get("parsed", {})
    judge_parse_error = bool(judge_data.get("parse_error"))
    judge_verdict = str(judge_parsed.get("verdict", "FAIL")).upper()
    judge_hallucination = judge_parsed.get("hallucination")
    no_context_case = is_no_context_case(case, context)
    no_context_answer_ok = is_expected_no_context_answer(generated_answer)

    heuristic_checks = {
        "expected_keywords_ok": expected_eval["score"] >= KEYWORD_PASS_THRESHOLD,
        "forbidden_keywords_ok": len(forbidden_eval["violations"]) == 0,
        "expected_answer_ok": True if exact_match is None else exact_match,
    }

    if no_context_case and no_context_answer_ok:
        judge_checks = {
            "judge_ok": True,
            "hallucination_ok": True,
        }
    else:
        judge_checks = {
            "judge_ok": True if judge_parse_error else judge_verdict == "PASS",
            "hallucination_ok": True if judge_parse_error else judge_hallucination is not True,
        }

    semantic_passed = all([
        heuristic_checks["expected_keywords_ok"],
        heuristic_checks["forbidden_keywords_ok"],
        heuristic_checks["expected_answer_ok"],
        judge_checks["judge_ok"],
        judge_checks["hallucination_ok"],
    ])
    strict_passed = semantic_passed and not judge_parse_error

    failure_reasons = []
    if not heuristic_checks["expected_keywords_ok"]:
        failure_reasons.append("missing_expected_keywords")
    if not heuristic_checks["forbidden_keywords_ok"]:
        failure_reasons.append("forbidden_keyword_found")
    if not heuristic_checks["expected_answer_ok"]:
        failure_reasons.append("expected_answer_mismatch")
    if not judge_checks["judge_ok"]:
        failure_reasons.append("judge_verdict_fail")
    if not judge_checks["hallucination_ok"]:
        failure_reasons.append("judge_marked_hallucination")
    if judge_parse_error:
        failure_reasons.append("judge_parse_error")

    return {
        "passed": semantic_passed,
        "strict_passed": strict_passed,
        "semantic_passed": semantic_passed,
        "failure_reasons": failure_reasons,
        "checks": {
            **heuristic_checks,
            **judge_checks,
            "judge_parse_ok": not judge_parse_error,
            "no_context_case": no_context_case,
            "no_context_answer_ok": no_context_answer_ok,
        },
        "keyword_metrics": expected_eval,
        "forbidden_metrics": forbidden_eval,
        "judge_metrics": {
            "relevance_score": judge_parsed.get("relevance_score"),
            "groundedness_score": judge_parsed.get("groundedness_score"),
            "terminology_score": judge_parsed.get("terminology_score"),
            "hallucination": judge_hallucination,
            "verdict": judge_parsed.get("verdict"),
            "analysis": judge_parsed.get("analysis", judge_data.get("cleaned_response", judge_data.get("raw_response", ""))),
            "parse_error": judge_parse_error,
        },
    }


def build_summary(results: List[Dict[str, Any]], started_at: float) -> Dict[str, Any]:
    total = len(results)
    semantic_passed = sum(1 for result in results if result["evaluation"]["semantic_passed"])
    strict_passed = sum(1 for result in results if result["evaluation"]["strict_passed"])
    semantic_failed = total - semantic_passed
    strict_failed = total - strict_passed

    total_llm_time = sum(result["generation"]["duration_sec"] for result in results)
    total_judge_time = sum(result["judge"]["duration_sec"] for result in results)

    avg_relevance = []
    avg_groundedness = []
    avg_terminology = []
    hallucination_count = 0
    parse_error_count = 0
    failure_reason_counts: Dict[str, int] = {}

    for result in results:
        metrics = result["evaluation"]["judge_metrics"]
        if metrics["relevance_score"] is not None:
            avg_relevance.append(metrics["relevance_score"])
        if metrics["groundedness_score"] is not None:
            avg_groundedness.append(metrics["groundedness_score"])
        if metrics["terminology_score"] is not None:
            avg_terminology.append(metrics["terminology_score"])
        if metrics["hallucination"] is True:
            hallucination_count += 1
        if metrics["parse_error"]:
            parse_error_count += 1
        for reason in result["evaluation"]["failure_reasons"]:
            failure_reason_counts[reason] = failure_reason_counts.get(reason, 0) + 1

    def average(values: List[Any]) -> float:
        return round(sum(values) / len(values), 2) if values else 0.0

    return {
        "total_cases": total,
        "semantic_passed": semantic_passed,
        "semantic_failed": semantic_failed,
        "semantic_pass_rate": round((semantic_passed / total) * 100, 1) if total else 0.0,
        "strict_passed": strict_passed,
        "strict_failed": strict_failed,
        "strict_pass_rate": round((strict_passed / total) * 100, 1) if total else 0.0,
        "hallucination_cases": hallucination_count,
        "judge_parse_errors": parse_error_count,
        "avg_relevance": average(avg_relevance),
        "avg_groundedness": average(avg_groundedness),
        "avg_terminology": average(avg_terminology),
        "total_generation_time_sec": round(total_llm_time, 3),
        "total_judge_time_sec": round(total_judge_time, 3),
        "total_runtime_sec": round(time.time() - started_at, 3),
        "failure_reasons": failure_reason_counts,
    }


def run_llm_test() -> int:
    dataset = load_dataset()
    if not dataset:
        print_step("Testandmed puuduvad. Lõpetan.")
        return 1

    run_started = time.time()
    existing_history = load_existing_log_history()
    log_data: Dict[str, Any] = {
        "run_started_at": ee_now_str(),
        "config": {
            "dataset_file": DATASET_FILE,
            "main_model": DEFAULT_MAIN_MODEL,
            "judge_model": DEFAULT_JUDGE_MODEL,
            "threads": DEFAULT_THREADS,
            "timeout": DEFAULT_TIMEOUT,
            "judge_timeout": DEFAULT_JUDGE_TIMEOUT,
            "keyword_pass_threshold": KEYWORD_PASS_THRESHOLD,
            "case_count": len(dataset),
        },
        "results": [],
        "summary": {},
    }
    print_step(f"ALUSTAN LLM TESTI ({len(dataset)} testilugu)")
    print_step(f"Põhimudel: {DEFAULT_MAIN_MODEL} | Hindajamudel: {DEFAULT_JUDGE_MODEL}")
    print_step(f"Märksõna lävend: {KEYWORD_PASS_THRESHOLD:.2f}")

    for index, case in enumerate(dataset, start=1):
        question = case.get("question", "")
        context = case.get("context", "")
        case_type = case.get("type", "rag")

        print_step(f"[{index}/{len(dataset)}] Küsimus: {question}")

        generation = generate_answer(
            question=question,
            context=context,
            model=DEFAULT_MAIN_MODEL,
            threads=DEFAULT_THREADS,
            timeout=DEFAULT_TIMEOUT,
        )
        print_step(
            f"[{index}/{len(dataset)}] Vastus valmis "
            f"({generation['duration_sec']:.2f}s, model_call={generation['used_model_call']})"
        )

        judge = judge_answer(
            question=question,
            context=context,
            answer=generation["answer"],
            expected_keywords=case.get("expected_keywords", []),
            forbidden_keywords=case.get("forbidden_keywords", []),
            model=DEFAULT_JUDGE_MODEL,
            threads=DEFAULT_THREADS,
            timeout=DEFAULT_JUDGE_TIMEOUT,
        )

        evaluation = evaluate_case(case, generation["answer"], judge)
        verdict = "PASS" if evaluation["semantic_passed"] else "FAIL"
        strict_marker = "strict=PASS" if evaluation["strict_passed"] else "strict=FAIL"

        print_step(
            f"[{index}/{len(dataset)}] {verdict} | {strict_marker} | "
            f"relevance={evaluation['judge_metrics']['relevance_score']} | "
            f"groundedness={evaluation['judge_metrics']['groundedness_score']} | "
            f"terminology={evaluation['judge_metrics']['terminology_score']} | "
            f"hallucination={evaluation['judge_metrics']['hallucination']}"
        )

        if evaluation["failure_reasons"]:
            print_step(f"[{index}/{len(dataset)}] põhjused: {', '.join(evaluation['failure_reasons'])}")

        case_result = {
            "index": index,
            "timestamp": ee_now_str(),
            "type": case_type,
            "question": question,
            "context": context,
            "expected_keywords": case.get("expected_keywords", []),
            "forbidden_keywords": case.get("forbidden_keywords", []),
            "expected_answer": case.get("expected_answer"),
            "generation": generation,
            "judge": judge,
            "evaluation": evaluation,
        }
        log_data["results"].append(case_result)
        log_data["summary"] = build_summary(log_data["results"], run_started)

    summary = build_summary(log_data["results"], run_started)
    log_data["summary"] = summary
    log_data["run_finished_at"] = ee_now_str()
    save_log(existing_history + [log_data])

    print("\n" + "=" * 72)
    print("LLM TESTI KOKKUVÕTE")
    print("-" * 72)
    print(f"Testilugusid kokku       : {summary['total_cases']}")
    print(f"Semantic PASS            : {summary['semantic_passed']}")
    print(f"Semantic FAIL            : {summary['semantic_failed']}")
    print(f"Semantic pass rate       : {summary['semantic_pass_rate']:.1f}%")
    print(f"Strict PASS              : {summary['strict_passed']}")
    print(f"Strict FAIL              : {summary['strict_failed']}")
    print(f"Strict pass rate         : {summary['strict_pass_rate']:.1f}%")
    print(f"Hallutsinatsiooniga juhud: {summary['hallucination_cases']}")
    print(f"Judge parse vigu         : {summary['judge_parse_errors']}")
    print(f"Keskmine asjakohasus     : {summary['avg_relevance']}/5")
    print(f"Keskmine korrektsus      : {summary['avg_groundedness']}/5")
    print(f"Keskmine terminoloogia   : {summary['avg_terminology']}/5")
    print(f"Genereerimise aeg kokku  : {summary['total_generation_time_sec']:.3f}s")
    print(f"Hindamise aeg kokku      : {summary['total_judge_time_sec']:.3f}s")
    print(f"Kogu jooksu aeg          : {summary['total_runtime_sec']:.3f}s")
    if summary["failure_reasons"]:
        print(f"Peamised põhjused        : {summary['failure_reasons']}")
    print(f"Logifail                 : {LOG_FILE}")
    print("=" * 72)

    return 0 if summary["semantic_failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(run_llm_test())
