import os
from datetime import datetime

from ingest_laws import (
    apply_source_metadata,
    get_law_sources,
    log_ingest_event,
    parse_xml_to_legal_chunks,
)
from oracle_ingest_common import (
    get_embedding_function,
    get_oracle_connection,
    write_document_chunks,
)


def run_ingest():
    start_time = datetime.now()
    print("\nORACLE LAWS INGEST")

    law_sources = []
    files_found = 0
    for source in get_law_sources():
        source_dir = source["dir"]
        if not os.path.exists(source_dir):
            print(f"WARNING: Kataloogi {source_dir} ei eksisteeri, jätan vahele.")
            continue
        files = sorted([f for f in os.listdir(source_dir) if f.lower().endswith(".akt")])
        files_found += len(files)
        law_sources.append((source, files))

    if not law_sources:
        print("ERROR: Ühtegi seaduste kataloogi ei leitud.")
        return

    log_ingest_event("SYSTEM", "ORACLE_LAW_IMPORT_STARTED", {"files_found": files_found})

    embed_fn = get_embedding_function()
    total_new_chunks = 0
    skipped_files = 0
    type_counts = {}

    with get_oracle_connection() as conn:
        cur = conn.cursor()
        for source, files in law_sources:
            source_dir = source["dir"]
            source_label = source["label"]
            classification_level = source["classification_level"]

            for filename in files:
                source_key = filename
                cur.execute("SELECT source_id FROM rag_sources WHERE source_key = :source_key", {"source_key": source_key})
                if cur.fetchone():
                    print(f"SKIP {filename} on juba olemas.")
                    skipped_files += 1
                    continue

                path = os.path.join(source_dir, filename)
                chunks, metas, doc_type = parse_xml_to_legal_chunks(path)
                metas = apply_source_metadata(metas, source_label, classification_level)

                if not chunks:
                    print(f"WARNING: Failis {filename} puudus sisu.")
                    continue

                class_note = f", classification={classification_level}" if classification_level else ""
                print(f"[{doc_type.upper()}] {filename} ({len(chunks)} osa{class_note})...", end=" ", flush=True)
                inserted = write_document_chunks(
                    cur,
                    embedding_fn=embed_fn,
                    source_key=source_key,
                    chunks=chunks,
                    metadatas=metas,
                )
                conn.commit()
                total_new_chunks += inserted
                type_counts[doc_type] = type_counts.get(doc_type, 0) + 1

                log_ingest_event(
                    doc_type,
                    "ORACLE_FILE_IMPORTED",
                    {
                        "file": filename,
                        "chunks": inserted,
                        "source_collection": source_label,
                        "classification_level": classification_level,
                    },
                )
                print("OK")

    duration = round((datetime.now() - start_time).total_seconds(), 2)
    log_ingest_event(
        "SYSTEM",
        "ORACLE_LAW_IMPORT_FINISHED",
        {
            "new_chunks": total_new_chunks,
            "duration_sec": duration,
            "types": type_counts,
            "skipped": skipped_files,
        },
    )
    print(f"\nIMPORT LOPPES. Kestus: {duration}s. Kokku uusi osi: {total_new_chunks}")


if __name__ == "__main__":
    run_ingest()
