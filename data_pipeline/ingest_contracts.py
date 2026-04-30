import os
import re
import json
import uuid
from datetime import datetime
from html.parser import HTMLParser

import chromadb
from chromadb.utils import embedding_functions

# --- KONFIGURATSIOON ---
CONTRACTS_DIR = "/app/storage/raw/contracts/"
DB_PATH = "/app/storage/vector_db"
INGEST_LOG_FILE = "/app/storage/ingest.log"
OLLAMA_URL = "http://ollama:11434"
EMBED_MODEL = "bge-m3"
MAX_CHUNK_CHARS = 900

# --- ANDMEBAASI ÜHENDUS ---
embedding_func = embedding_functions.OllamaEmbeddingFunction(
    model_name=EMBED_MODEL,
    url=OLLAMA_URL,
)

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(
    name="procurements",
    embedding_function=embedding_func,
)


def log_ingest_event(doc_type, event_name, details):
    """Salvestab impordi sündmused JSON logifaili."""
    try:
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "doc_type": doc_type,
            "event": event_name,
            "details": details,
        }
        with open(INGEST_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"VIGA LOGIMISEL: {e}")


def normalize_text(text):
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text


def smart_truncate(text, max_chars=MAX_CHUNK_CHARS):
    text = normalize_text(text)
    if len(text) <= max_chars:
        return text

    cutoff = text[: max_chars + 1]
    preferred_breaks = [cutoff.rfind(mark) for mark in [". ", "; ", ": ", "! ", "? "]]
    preferred_break = max(preferred_breaks)
    if preferred_break >= int(max_chars * 0.7):
        return cutoff[: preferred_break + 1].rstrip()

    word_break = cutoff.rfind(" ")
    if word_break >= int(max_chars * 0.7):
        return cutoff[:word_break].rstrip() + "..."

    return cutoff[:max_chars].rstrip() + "..."


def is_estonian_personal_code(value):
    """Kontrollib Eesti isikukoodi kuju ja kontrollnumbrit."""
    value = str(value or "").strip()
    if not re.fullmatch(r"[1-8]\d{10}", value):
        return False

    weights_1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1]
    weights_2 = [3, 4, 5, 6, 7, 8, 9, 1, 2, 3]
    digits = [int(ch) for ch in value]

    check = sum(digits[i] * weights_1[i] for i in range(10)) % 11
    if check == 10:
        check = sum(digits[i] * weights_2[i] for i in range(10)) % 11
    if check == 10:
        check = 0
    return check == digits[10]


class ContractHtmlParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.meta = {}
        self.title = ""
        self.sections = []
        self.current_heading = ""
        self.current_parts = []
        self.capture_title = False
        self.capture_heading = False
        self.heading_parts = []
        self.capture_text = False
        self.text_parts = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag == "meta" and attrs_dict.get("name"):
            self.meta[attrs_dict["name"]] = attrs_dict.get("content", "")
        elif tag == "title":
            self.capture_title = True
            self.text_parts = []
        elif tag in {"h1", "h2"}:
            self._flush_section()
            self.capture_heading = True
            self.heading_parts = []
        elif tag in {"p", "li"}:
            self.capture_text = True
            self.text_parts = []

    def handle_endtag(self, tag):
        if tag == "title" and self.capture_title:
            self.title = normalize_text(" ".join(self.text_parts))
            self.capture_title = False
            self.text_parts = []
        elif tag in {"h1", "h2"} and self.capture_heading:
            self.current_heading = normalize_text(" ".join(self.heading_parts))
            self.capture_heading = False
            self.heading_parts = []
        elif tag in {"p", "li"} and self.capture_text:
            text = normalize_text(" ".join(self.text_parts))
            if text:
                self.current_parts.append(text)
            self.capture_text = False
            self.text_parts = []

    def handle_data(self, data):
        text = data.strip()
        if not text:
            return
        if self.capture_title:
            self.text_parts.append(text)
        elif self.capture_heading:
            self.heading_parts.append(text)
        elif self.capture_text:
            self.text_parts.append(text)

    def close(self):
        super().close()
        self._flush_section()

    def _flush_section(self):
        if self.current_heading or self.current_parts:
            section_text = normalize_text(" ".join(self.current_parts))
            if section_text:
                self.sections.append(
                    {
                        "heading": self.current_heading or self.title or "Leping",
                        "text": section_text,
                    }
                )
        self.current_heading = ""
        self.current_parts = []


def parse_contract_html(file_path):
    filename = os.path.basename(file_path)
    with open(file_path, "r", encoding="utf-8-sig") as f:
        html = f.read()

    parser = ContractHtmlParser()
    parser.feed(html)
    parser.close()

    meta = {str(k): normalize_text(v) for k, v in parser.meta.items()}
    title = parser.title or meta.get("contract_id") or filename
    counterparty_type = meta.get("counterparty_type", "")
    subject_id = meta.get("subject_id", "")
    contains_personal_data = (
        counterparty_type == "private_person"
        or is_estonian_personal_code(subject_id)
        or bool(re.search(r"\b[1-8]\d{10}\b", html))
    )

    base_meta = {
        "doc_type": meta.get("document_type", "contract"),
        "type": "leping",
        "file": filename,
        "source": meta.get("contract_id", filename),
        "contract_id": meta.get("contract_id", ""),
        "classification_level": meta.get("classification_level", "secret") or "secret",
        "tenant_id": meta.get("tenant_id", ""),
        "tenant_name": meta.get("tenant_name", ""),
        "subject_id": subject_id,
        "subject_name": meta.get("subject_name", ""),
        "counterparty_id": meta.get("counterparty_id", subject_id),
        "counterparty_type": counterparty_type,
        "contains_personal_data": contains_personal_data,
        "title": title,
    }

    chunks = []
    metadatas = []
    for index, section in enumerate(parser.sections, start=1):
        heading = section["heading"]
        body = section["text"]
        if not body:
            continue

        display_name = f"{base_meta['contract_id']} | {heading}" if base_meta["contract_id"] else heading
        chunk_text = smart_truncate(
            f"[LEPING] {display_name}: {body}",
            MAX_CHUNK_CHARS,
        )
        chunk_meta = dict(base_meta)
        chunk_meta.update(
            {
                "chunk_type": "contract_section",
                "section_index": index,
                "section_title": heading[:250],
                "display_name": display_name[:250],
            }
        )
        chunks.append(chunk_text)
        metadatas.append(chunk_meta)

    return chunks, metadatas


def run_ingest():
    start_time = datetime.now()
    print(f"\n📄 CONTRACT INGEST | Mudel: {EMBED_MODEL}")

    if not os.path.exists(CONTRACTS_DIR):
        print(f"❌ VIGA: Kataloogi {CONTRACTS_DIR} ei eksisteeri!")
        return

    files = sorted([f for f in os.listdir(CONTRACTS_DIR) if f.lower().endswith((".html", ".htm"))])
    log_ingest_event("SYSTEM", "CONTRACT_IMPORT_STARTED", {"files_found": len(files)})

    total_new_chunks = 0
    skipped_files = 0
    batch_size = 10

    for filename in files:
        existing = collection.get(
            where={
                "$and": [
                    {"file": filename},
                    {"doc_type": "contract"},
                ]
            },
            limit=1,
        )
        if existing and existing["ids"]:
            print(f"⏩ {filename} on juba olemas.")
            skipped_files += 1
            continue

        path = os.path.join(CONTRACTS_DIR, filename)
        try:
            chunks, metadatas = parse_contract_html(path)
        except Exception as e:
            print(f"❌ {filename} parsimine ebaõnnestus: {e}")
            log_ingest_event("contract", "CONTRACT_IMPORT_FAILED", {"file": filename, "error": str(e)})
            continue

        if not chunks:
            print(f"⚠️ Failis {filename} puudus sisu.")
            continue

        contract_id = metadatas[0].get("contract_id", filename)
        subject_name = metadatas[0].get("subject_name", "")
        print(f"📥 [LEPING] {contract_id} | {subject_name} ({len(chunks)} osa)...", end=" ", flush=True)

        for i in range(0, len(chunks), batch_size):
            end = i + batch_size
            collection.add(
                documents=chunks[i:end],
                ids=[f"contract-{uuid.uuid4()}" for _ in chunks[i:end]],
                metadatas=metadatas[i:end],
            )

        total_new_chunks += len(chunks)
        log_ingest_event(
            "contract",
            "CONTRACT_IMPORTED",
            {
                "file": filename,
                "contract_id": contract_id,
                "subject_id": metadatas[0].get("subject_id", ""),
                "tenant_id": metadatas[0].get("tenant_id", ""),
                "chunks": len(chunks),
            },
        )
        print("✅")

    duration = round((datetime.now() - start_time).total_seconds(), 2)
    log_ingest_event(
        "SYSTEM",
        "CONTRACT_IMPORT_FINISHED",
        {
            "new_chunks": total_new_chunks,
            "duration_sec": duration,
            "skipped": skipped_files,
        },
    )
    print(f"\n✨ LEPINGUTE IMPORT LÕPPES. Kestus: {duration}s. Kokku uusi osi: {total_new_chunks}")


if __name__ == "__main__":
    run_ingest()
