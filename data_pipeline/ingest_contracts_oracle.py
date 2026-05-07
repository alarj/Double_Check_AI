import os
from datetime import datetime

from ingest_contracts import log_ingest_event, parse_contract_html
from oracle_ingest_common import (
    get_embedding_function,
    get_oracle_connection,
    write_document_chunks,
)


CONTRACTS_DIR = "/app/storage/raw/contracts/"


def run_ingest():
    start_time = datetime.now()
    print("\nORACLE CONTRACT INGEST")

    if not os.path.exists(CONTRACTS_DIR):
        print(f"ERROR: Kataloogi {CONTRACTS_DIR} ei eksisteeri!")
        return

    files = sorted([f for f in os.listdir(CONTRACTS_DIR) if f.lower().endswith((".html", ".htm"))])
    log_ingest_event("SYSTEM", "ORACLE_CONTRACT_IMPORT_STARTED", {"files_found": len(files)})

    embed_fn = get_embedding_function()
    total_new_chunks = 0
    skipped_files = 0

    with get_oracle_connection() as conn:
        cur = conn.cursor()
        for filename in files:
            source_key = filename
            cur.execute("SELECT source_id FROM rag_sources WHERE source_key = :source_key", {"source_key": source_key})
            if cur.fetchone():
                print(f"SKIP {filename} on juba olemas.")
                skipped_files += 1
                continue

            path = os.path.join(CONTRACTS_DIR, filename)
            try:
                chunks, metadatas = parse_contract_html(path)
            except Exception as e:
                print(f"ERROR: {filename} parsimine ebaonnestus: {e}")
                log_ingest_event("contract", "ORACLE_CONTRACT_IMPORT_FAILED", {"file": filename, "error": str(e)})
                continue

            if not chunks:
                print(f"WARNING: Failis {filename} puudus sisu.")
                continue

            contract_id = metadatas[0].get("contract_id", filename)
            subject_name = metadatas[0].get("subject_name", "")
            print(f"[LEPING] {contract_id} | {subject_name} ({len(chunks)} osa)...", end=" ", flush=True)

            inserted = write_document_chunks(
                cur,
                embedding_fn=embed_fn,
                source_key=source_key,
                chunks=chunks,
                metadatas=metadatas,
            )
            conn.commit()
            total_new_chunks += inserted

            log_ingest_event(
                "contract",
                "ORACLE_CONTRACT_IMPORTED",
                {
                    "file": filename,
                    "contract_id": contract_id,
                    "subject_id": metadatas[0].get("subject_id", ""),
                    "tenant_id": metadatas[0].get("tenant_id", ""),
                    "chunks": inserted,
                },
            )
            print("OK")

    duration = round((datetime.now() - start_time).total_seconds(), 2)
    log_ingest_event(
        "SYSTEM",
        "ORACLE_CONTRACT_IMPORT_FINISHED",
        {
            "new_chunks": total_new_chunks,
            "duration_sec": duration,
            "skipped": skipped_files,
        },
    )
    print(f"\nLEPINGUTE IMPORT LOPPES. Kestus: {duration}s. Kokku uusi osi: {total_new_chunks}")


if __name__ == "__main__":
    run_ingest()
