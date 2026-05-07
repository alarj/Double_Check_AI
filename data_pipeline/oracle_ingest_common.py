import json
import os
from typing import Dict, Iterable, List, Optional

import oracledb
from chromadb.utils import embedding_functions


ORACLE_EMBED_MODEL = os.getenv("ORACLE_EMBED_MODEL", "bge-m3")
ORACLE_OLLAMA_URL = os.getenv("ORACLE_OLLAMA_URL", "http://ollama:11434")
ORACLE_EMBED_BATCH_SIZE = int(os.getenv("ORACLE_EMBED_BATCH_SIZE", "32"))


def get_embedding_function():
    return embedding_functions.OllamaEmbeddingFunction(
        model_name=ORACLE_EMBED_MODEL,
        url=ORACLE_OLLAMA_URL,
    )


def get_oracle_connection():
    oracle_dsn = os.getenv("ORACLE_DSN", "").strip()
    oracle_user = os.getenv("ORACLE_USER", "").strip()
    oracle_password = os.getenv("ORACLE_PASSWORD", "").strip()
    oracle_config_dir = os.path.expanduser(os.getenv("ORACLE_CONFIG_DIR", "").strip())
    oracle_wallet_location = os.path.expanduser(os.getenv("ORACLE_WALLET_LOCATION", "").strip())
    oracle_wallet_password = os.getenv("ORACLE_WALLET_PASSWORD", "").strip()

    if not oracle_dsn or not oracle_user or not oracle_password:
        raise RuntimeError("Missing Oracle connection env vars: ORACLE_DSN, ORACLE_USER, ORACLE_PASSWORD")

    connect_kwargs = {
        "user": oracle_user,
        "password": oracle_password,
        "dsn": oracle_dsn,
    }

    # Wallet-based mTLS setup for Oracle Autonomous DB.
    if oracle_config_dir:
        connect_kwargs["config_dir"] = oracle_config_dir
    if oracle_wallet_location:
        connect_kwargs["wallet_location"] = oracle_wallet_location
    if oracle_wallet_password:
        connect_kwargs["wallet_password"] = oracle_wallet_password

    return oracledb.connect(**connect_kwargs)


def _json_dumps(value) -> str:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _source_type_from_meta(meta: Dict) -> str:
    if str(meta.get("doc_type", "")).lower() == "contract":
        return "contract"
    if meta.get("law"):
        return "law"
    return "other"


def upsert_source(cur, source_key: str, meta: Dict) -> int:
    source_type = _source_type_from_meta(meta)
    source_file = str(meta.get("file") or source_key)[:500]
    title = str(meta.get("title") or meta.get("display_name") or source_key)[:500]
    doc_type = str(meta.get("doc_type") or meta.get("type") or "")[:50] or None
    tenant_id = str(meta.get("tenant_id") or "")[:100] or None
    classification_level = str(meta.get("classification_level") or "public")[:30]
    payload_json = _json_dumps(meta)

    cur.execute(
        """
        MERGE INTO rag_sources s
        USING (SELECT :source_key AS source_key FROM dual) q
        ON (s.source_key = q.source_key)
        WHEN MATCHED THEN UPDATE SET
            source_type = :source_type,
            source_file = :source_file,
            title = :title,
            doc_type = :doc_type,
            tenant_id = :tenant_id,
            classification_level = :classification_level,
            payload_json = :payload_json,
            updated_at = SYSTIMESTAMP
        WHEN NOT MATCHED THEN INSERT (
            source_type, source_key, source_file, title, doc_type, tenant_id, classification_level, payload_json
        ) VALUES (
            :source_type, :source_key, :source_file, :title, :doc_type, :tenant_id, :classification_level, :payload_json
        )
        """,
        {
            "source_type": source_type,
            "source_key": source_key,
            "source_file": source_file,
            "title": title,
            "doc_type": doc_type,
            "tenant_id": tenant_id,
            "classification_level": classification_level,
            "payload_json": payload_json,
        },
    )

    cur.execute("SELECT source_id FROM rag_sources WHERE source_key = :source_key", {"source_key": source_key})
    row = cur.fetchone()
    if not row:
        raise RuntimeError(f"Failed to resolve source_id for source_key={source_key}")
    return int(row[0])


def upsert_chunk(cur, source_id: int, chunk_uid: str, chunk_index: int, chunk_text: str, meta: Dict) -> int:
    chunk_type = str(meta.get("chunk_type") or "")[:50] or None
    section_title = str(meta.get("section_title") or meta.get("paragraph_title") or "")[:500] or None
    metadata_json = _json_dumps(meta)

    cur.execute(
        """
        MERGE INTO rag_chunks c
        USING (SELECT :chunk_uid AS chunk_uid FROM dual) q
        ON (c.chunk_uid = q.chunk_uid)
        WHEN MATCHED THEN UPDATE SET
            source_id = :source_id,
            chunk_index = :chunk_index,
            chunk_type = :chunk_type,
            section_title = :section_title,
            content = :content,
            metadata_json = :metadata_json,
            updated_at = SYSTIMESTAMP
        WHEN NOT MATCHED THEN INSERT (
            source_id, chunk_uid, chunk_index, chunk_type, section_title, content, metadata_json
        ) VALUES (
            :source_id, :chunk_uid, :chunk_index, :chunk_type, :section_title, :content, :metadata_json
        )
        """,
        {
            "source_id": source_id,
            "chunk_uid": chunk_uid,
            "chunk_index": int(chunk_index),
            "chunk_type": chunk_type,
            "section_title": section_title,
            "content": chunk_text,
            "metadata_json": metadata_json,
        },
    )

    cur.execute("SELECT chunk_id FROM rag_chunks WHERE chunk_uid = :chunk_uid", {"chunk_uid": chunk_uid})
    row = cur.fetchone()
    if not row:
        raise RuntimeError(f"Failed to resolve chunk_id for chunk_uid={chunk_uid}")
    return int(row[0])


def upsert_embedding(cur, chunk_id: int, embedding_model: str, embedding: List[float]):
    embedding_json = _json_dumps(embedding)
    cur.execute(
        """
        MERGE INTO rag_embeddings e
        USING (
            SELECT :chunk_id AS chunk_id, :embedding_model AS embedding_model FROM dual
        ) q
        ON (e.chunk_id = q.chunk_id AND e.embedding_model = q.embedding_model)
        WHEN MATCHED THEN UPDATE SET
            embedding_dim = 1024,
            embedding = TO_VECTOR(:embedding_json),
            created_at = SYSTIMESTAMP
        WHEN NOT MATCHED THEN INSERT (
            chunk_id, embedding_model, embedding_dim, embedding
        ) VALUES (
            :chunk_id, :embedding_model, 1024, TO_VECTOR(:embedding_json)
        )
        """,
        {
            "chunk_id": int(chunk_id),
            "embedding_model": embedding_model,
            "embedding_json": embedding_json,
        },
    )


def write_document_chunks(
    cur,
    embedding_fn,
    source_key: str,
    chunks: Iterable[str],
    metadatas: Iterable[Dict],
    embedding_model: Optional[str] = None,
):
    metas = list(metadatas)
    docs = list(chunks)
    if not docs:
        return 0

    source_meta = metas[0] if metas else {}
    source_id = upsert_source(cur, source_key=source_key, meta=source_meta)
    model_name = embedding_model or ORACLE_EMBED_MODEL

    batch_size = max(1, ORACLE_EMBED_BATCH_SIZE)
    for start in range(0, len(docs), batch_size):
        end = min(start + batch_size, len(docs))
        batch_docs = docs[start:end]
        batch_metas = metas[start:end]
        batch_vectors = embedding_fn(batch_docs)

        for offset, (text, meta, vec) in enumerate(zip(batch_docs, batch_metas, batch_vectors)):
            idx = start + offset
            chunk_uid = f"{source_key}_{idx}"
            chunk_id = upsert_chunk(
                cur,
                source_id=source_id,
                chunk_uid=chunk_uid,
                chunk_index=idx,
                chunk_text=text,
                meta=meta or {},
            )
            upsert_embedding(cur, chunk_id=chunk_id, embedding_model=model_name, embedding=vec)

    return len(docs)
