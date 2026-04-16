import time
import json
import os
import chromadb
import uuid
import sys
import shutil
import xml.etree.ElementTree as ET
import re
from chromadb.utils import embedding_functions

# --- SEADISTUSED ---
MODELS = ["mxbai-embed-large", "nomic-embed-text", "bge-m3"]
LAWS_DIR = "/app/storage/raw/laws/"
DB_BASE_PATH = "/app/storage/vector_db_bench"
DATASET_PATH = "/app/data_pipeline/eval_dataset.json"
OLLAMA_URL = "http://ollama:11434"
TOP_K = 5
MAX_CHUNK_SIZE = 450  # Veelgi konservatiivsem limiit, et vältida 400 vigu

def strip_ns(tag):
    """Eemaldab XML nimeruumi."""
    return tag.split('}')[-1] if '}' in tag else tag

def split_text_smart(text, max_size, section_prefix=""):
    """Tükeldab teksti targalt lausete kaupa, säilitades eesliite."""
    if not text: return []
    full_prefix = f"{section_prefix}: " if section_prefix else ""
    
    # Kui tekst koos eesliitega on juba väike
    if len(full_prefix + text) <= max_size:
        return [full_prefix + text]
    
    chunks = []
    # Jagame lauseteks (punkt, küsimärk, hüüumärk millele järgneb tühik)
    sentences = re.split(r'(?<=[.!?]) +', text)
    current_content = ""
    
    for sentence in sentences:
        # Kontrollime pikkust
        if len(full_prefix + current_content + sentence) <= max_size:
            current_content += (" " + sentence if current_content else sentence)
        else:
            if current_content:
                chunks.append(full_prefix + current_content.strip())
            current_content = sentence
            # Kui üks lause on ikka liiga pikk, siis lõikame jõuga
            if len(full_prefix + current_content) > max_size:
                chunks.append((full_prefix + current_content)[:max_size])
                current_content = ""
                
    if current_content:
        chunks.append(full_prefix + current_content.strip())
    return chunks

def robust_parse_xml(file_path):
    """Parsib seaduse XML faili ja otsib paragrahve või muud sisu."""
    chunks = []
    metas = []
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        law_abbr = "AKT"
        for l_elem in root.iter():
            if strip_ns(l_elem.tag) == "lyhend":
                law_abbr = l_elem.text if l_elem.text else "AKT"
                break

        found_paras = 0
        for para in root.iter():
            if strip_ns(para.tag) == "paragrahv":
                found_paras += 1
                para_nr = ""
                para_content = []
                
                # Kogume numbri ja sisu
                for child in para.iter():
                    tag = strip_ns(child.tag)
                    if tag == "paragrahvNr":
                        para_nr = child.text if child.text else ""
                    elif tag in ["sisuTekst", "tavatekst", "alapunkt"]:
                        txt = "".join(child.itertext()).strip()
                        if txt: para_content.append(txt)
                
                if para_content:
                    full_text = " ".join(para_content)
                    prefix = f"{law_abbr} § {para_nr}"
                    # Loome tükid
                    sub_chunks = split_text_smart(full_text, MAX_CHUNK_SIZE, section_prefix=prefix)
                    for sc in sub_chunks:
                        chunks.append(sc)
                        metas.append({"section": str(para_nr), "source": prefix})
        
        # Kui paragrahve polnud, proovime üldist teksti
        if found_paras == 0:
            for elem in root.iter():
                if strip_ns(elem.tag) in ["sisu", "tekst"]:
                    txt = "".join(elem.itertext()).strip()
                    if len(txt) > 100:
                        sub_chunks = split_text_smart(txt, MAX_CHUNK_SIZE, section_prefix=f"{law_abbr} Üld")
                        for sc in sub_chunks:
                            chunks.append(sc)
                            metas.append({"section": "0", "source": law_abbr})
                            
        return chunks, metas
    except Exception as e:
        print(f"    ❌ VIGA XML parsimisel: {e}")
        return [], []

def evaluate_model(model_name, dataset):
    print(f"\n" + "="*50)
    print(f"🚀 TESTIN MUDELIT: {model_name}")
    print("="*50)
    
    # --- PUHAS LEHT: Kustutame vana andmebaasi kataloogi täielikult ---
    db_path = f"{DB_BASE_PATH}_{model_name}"
    if os.path.exists(db_path):
        print(f"  🧹 Kustutan vana baasi: {db_path}")
        shutil.rmtree(db_path)
    
    try:
        # Algatame Chroma ja embedding funktsiooni
        ef = embedding_functions.OllamaEmbeddingFunction(model_name=model_name, url=OLLAMA_URL)
        client = chromadb.PersistentClient(path=db_path)
        collection = client.create_collection(name="bench_coll", embedding_function=ef)

        # Kontrollime seaduste failide olemasolu
        files = [f for f in os.listdir(LAWS_DIR) if f.lower().endswith(".akt")]
        if not files:
            print(f"  ❌ VIGA: Seaduste kataloog {LAWS_DIR} on tühi!")
            return None

        print(f"  📥 Laadin andmed ({len(files)} faili)...")
        total_chunks = 0

        for filename in files:
            path = os.path.join(LAWS_DIR, filename)
            chunks, metas = robust_parse_xml(path)
            
            if chunks:
                # Lisame andmed andmebaasi (batch kaupa, et vältida timeouti)
                batch_size = 10
                for i in range(0, len(chunks), batch_size):
                    end = i + batch_size
                    collection.add(
                        documents=chunks[i:end],
                        ids=[f"{model_name}-{uuid.uuid4()}" for _ in chunks[i:end]],
                        metadatas=metas[i:end]
                    )
                total_chunks += len(chunks)
                print(f"    ✅ {filename}: {len(chunks)} tükki laaditud.")

        if total_chunks == 0:
            print("  ❌ VIGA: Ühtegi andmetükki ei laaditud!")
            return None

        # --- TESTIMINE ---
        print(f"  🔍 Alustan testimist ({len(dataset)} küsimust)...")
        stats = {"top1": 0, "top5": 0, "ranks": [], "latency": []}

        for item in dataset:
            q = item["question"]
            expected = str(item["expected_section"])
            
            start_t = time.time()
            results = collection.query(query_texts=[q], n_results=TOP_K)
            stats["latency"].append(time.time() - start_t)

            metas_found = results.get("metadatas", [[]])[0]
            found_sections = [str(m.get("section")) for m in metas_found]
            
            rank = -1
            for i, sec in enumerate(found_sections):
                if sec == expected:
                    rank = i + 1
                    break
            
            if rank == 1: stats["top1"] += 1
            if rank != -1:
                stats["top5"] += 1
                stats["ranks"].append(rank)
                print(f"    ❓ '{q[:40]}...' -> ✅ Leitud (§{expected}) positsioonil {rank}")
            else:
                stats["ranks"].append(TOP_K + 1)
                print(f"    ❓ '{q[:40]}...' -> ❌ EI LEITUD (§{expected})")

        return {
            "model": model_name,
            "acc_top1": (stats["top1"] / len(dataset)) * 100,
            "acc_top5": (stats["top5"] / len(dataset)) * 100,
            "avg_rank": sum(stats["ranks"]) / len(stats["ranks"]),
            "avg_lat": sum(stats["latency"]) / len(stats["latency"])
        }

    except Exception as e:
        print(f"  ❌ KRIITILINE VIGA: {e}")
        return None

if __name__ == "__main__":
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
        
    all_results = []
    for model in MODELS:
        res = evaluate_model(model, eval_data)
        if res:
            all_results.append(res)

    if all_results:
        print("\n" + "="*85)
        print(f"{'MUDEL':<22} | {'TOP 1':<10} | {'TOP 5':<10} | {'KESKM. KOHT':<12} | {'LATENTSUS'}")
        print("-" * 85)
        for r in all_results:
            print(f"{r['model']:<22} | {r['acc_top1']:>8.1f}% | {r['acc_top5']:>8.1f}% | {r['avg_rank']:>11.2f} | {r['avg_lat']:>7.3f}s")
        print("="*85 + "\n")