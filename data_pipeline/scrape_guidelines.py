import requests
from bs4 import BeautifulSoup
import chromadb
from chromadb.utils import embedding_functions
import os

# Konfiguratsioon
URL = "https://fin.ee/riigihanked-riigiabi-osalused/riigihanked/korduma-kippuvad-kusimused"
DB_PATH = "../storage/vector_db"
OLLAMA_URL = "http://localhost:11434/api/embeddings"

def scrape_guidelines():
    print(f"--- Alustan veebilehe skreipimist: {URL} ---")
    
    try:
        response = requests.get(URL, timeout=10)
        response.raise_for_status()
        response.encoding = 'utf-8'
    except Exception as e:
        print(f"Viga lehe laadimisel: {e}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Rahandusministeeriumi lehel on sisu tavaliselt 'node__content' või 'field-items' sees
    # Otsime peamist artikli osa
    content_area = soup.find('div', class_='node__content') or soup.find('article')
    
    if not content_area:
        print("Viga: Ei leidnud lehelt sisuosa. Kontrolli HTML struktuuri.")
        return

    # Ühendume vektorbaasiga
    client = chromadb.PersistentClient(path=DB_PATH)
    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        url=OLLAMA_URL,
        model_name="mxbai-embed-large"
    )
    collection = client.get_or_create_collection(
        name="procurements", 
        embedding_function=ollama_ef
    )

    # Leiame kõik küsimused (tavaliselt <strong> või pealkirjad) ja vastused
    # Lihtsustatud lähenemine: võtame lõigud ja tükeldame need mõistlikeks osadeks
    paragraphs = content_area.find_all(['p', 'li', 'h3'])
    
    count = 0
    current_chunk = ""
    
    for element in paragraphs:
        text = element.get_text().strip()
        if not text:
            continue
            
        # Kui tekst on piisavalt pikk, lisame selle baasi
        # Võime ka teksti akumuleerida, et tekiksid terviklikumad mõttepausid
        if len(text) > 40:
            collection.add(
                documents=[text],
                metadatas=[{
                    "source": URL,
                    "type": "guideline",
                    "title": "Rahandusministeeriumi KKK"
                }],
                ids=[f"guideline_{count}_{os.urandom(4).hex()}"]
            )
            count += 1

    print(f"--- Valmis! Lisati {count} teadmuskifdu vektorbaasi. ---")

if __name__ == "__main__":
    scrape_guidelines()