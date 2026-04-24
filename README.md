# Double_Check_AI

Tegemist on TalTech AI mikrokraadi kursusetööga.
Võid vabalt lugeda, kopeerida, endale paigaldada jne. 
Mõnes oma kirjalikus töös või ettekandes kasutamisel on viisakas viidata :-)

# Double_Check_AI: AI Guardrail Prototüüp

See projekt demonstreerib mitmeastmelist AI turvakihti (Guardrail), mis kasutab mitut erinevat mudelit, et tagada siseandmete turvalisus.

## Projekti struktuur

* **`logic/`** - Rakenduse tuumik.
    * `main.py` - Streamliti veebiliides ja AI loogika.
    * `Dockerfile` - Juhised Pythoni keskkonna ja sõltuvuste seadistamiseks.
    * `requirements.txt` - Vajalikud Pythoni paketid (Streamlit, Requests jne).
* **`data_pipeline/`** - Andmete ettevalmistus.
    * `ingest.py` - Skript XML failide töötlemiseks ja vektorbaasi saatmiseks.
    * `parsers.py` - Loogika konkreetsete hanke-XML struktuuride lugemiseks.
* **`storage/`** - (Kohalik andmehoidla, ignoreeritud Gitis)
    * `raw/procurements/` - Koht, kuhu kasutaja paneb oma XML failid.
    * `vector_db/` - ChromaDB poolt genereeritud vektorandmebaas.
* **`docker-compose.yml`** - Orkestreerib kahte teenust: Ollama (AI mootor) ja Pythoni rakendus.
* **`.gitignore`** - Välistab andmete, virtuaalkeskkonna ja mudelite sattumise Giti.

## Kuidas käivitada

1.  **Klooni projekt ja seadista keskkond:**
    ```bash
    git clone https://github.com/alarj/Double_Check_AI.git
    cd Double_Check_AI
    git checkout main
    ```
	**Kui soovid, siis loo ja aktiveeri virtuaalkeskkond**
    Kuna kogu rakendus töötab dockeri konteinerites, siis venv osa võib vahele jätta, seda pole reaalselt vaja.
	```bash
	python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
	```
	
2.  **Installi docker ja docker compose**
    ```bash
    sudo apt update
    sudo apt install docker.io
    sudo apt install docker-compose-v2 -y
    ```
    Anna oma kasutajale kohe ka õigused, et dockerit ei peaks sudo kaudu iga kord käivitama
     ```bash
    sudo usermod -aG docker $USER
    ```
    Selleks, et ei peaks õiguste muutmiseks välja logima
    ```bash
    newgrp docker
    ```
3.  **Käivita Docker Compose:**
    ```bash
    docker compose up -d --build
    ```
    **NB!** Kui käsk docker compose up lõpetab, kirjuta ```docker ps```, et näha, kas rakendus on "Up" staatuses ja millist porti ta kasutab!

4.  **Laadi alla AI mudelid (optional):**
    Esmakordsel seadistamisel võib mudelid käsitsi Ollama konteinerisse tõmmata, järgmistel käivitamistel pole seda enam vaja
	Kõik vajalikud mudelid on kirjeldatud ka docker-compose.yml failis, seega laeb süsteem need ise konteineri ehitamise käigus alla kui vaja.
	Mudelite täielik loetelu on allpool Tehnoloogiad punktis

	**Vaikimisi eeldatakse**
	
	Põhimudelid:
    ```bash
    docker exec -it ollama ollama pull gemma2:2b
    docker exec -it ollama ollama pull llama3:8b
    ``` 
	Embedding mudel RAG jaoks:
	```bash
    docker exec -it ollama ollama pull bge-m3
    ``` 
	
5.  **Ava rakendus:**
    Liides on kättesaadav aadressil `http://<serveri-ip>:8501` (serveri IP-aadressil pordis 8501).
	REST API swagger on kättesaadav `http://<serveri-ip>:8000/swagger` (serveri IP-aadressil pordis 8000)
    **NB!** serveri port 8501 ja 8000 peavad olema internetist kättesaadavad. Võib nõuda serveri või teenuspakkuja keskkonna eraldi häälestamist turvareeglite osas.

## Andmete ettevalmistus (RAG)

Süsteem kasutab RAG-loogikat (Retrieval-Augmented Generation). Enne rakenduse kasutamist tuleb andmed vektorbaasi laadida. 
**NB!! Hetkel on andmed gitis olemas, seega pole neid igal serveri uuendamisel vaja lisada.**
Andmeid saab sisse logeda algandmetest, kui seda on vaja teha, siis tee nii:

### Eeltingimused
- Ollama peab olema installitud ja jooksma.

1.  **Kopeeri andmed**
    
	Pane seaduste failid storage/raw/laws/
	
	
2.  **Käivita indekseerimine**
    *Hanked (seda pole vaja hetkel, ära kasuta):*
	```bash
    source .venv/bin/activate
    cd data_pipeline
    python3 ingest.py
    ``` 
	*Seadused (see töötab ja on testitud):*
	```bash
    docker exec -it logic-app python /app/data_pipeline/ingest_laws.py
     ``` 
	*Veebi juhendid: (seda pole vaja hetkel, ära kasuta)*
	```bash
    source .venv/bin/activate
    cd data_pipeline
    python3 scrape_guidelines.py
    ``` 

## Tehnoloogiad
* **Streamlit** - Veebiliides.
* **Ollama** - Lokaalne AI mudelite serveerimine.
* **llama3:8b** - Põhiline vastuste genereerimise mudel.
* **gemma2:2b** - Kergekaaluline ja kiire kontrollmudel (Guardrail).
* **phi3** - Täiendav kontrollmudel testimiseks/benchmarkiks.
* **mistral** - Täiendav kontrollmudel testimiseks/benchmarkiks.
* **bge-m3** - Aktiivselt kasutatav embedding-mudel RAG jaoks.
* **alarjoeste/estonian-normalizer** - Eesti keele normaliseerimise mudel.
* **mxbai-embed-large** - Embedding-mudel, mis laaditakse keskkonda, kuid hetkel aktiivses töövoos ei kasutata.

## Turvakihi loogika
Rakendus oskab kasutada kuni 3 astmelist protsessi:
1. **Eelkontroll**  analüüsib sisendit. Kui see on ohtlik, päringut põhimudelile ei saadeta. Vajadusel üritab kasutaja sisendit normaliseerida.
2. **Põhipäring**  genereerib vastuse
3. **Järelkontroll** kontrollib väljundit, et vältida tundliku info leket või ebasobivat sisu.

## Estonian normaliseerija mudel 
Normaliseerimise üks variant on kasutada mõnda hästi eesti keelt valdavat mudelit, mitte valmis mudeleid.
Tiimiliikmed ei pea mudelit ise GGUF-failist looma ega hoidma oma serveris `models/` kaustas suuri mudelifaile.
Kasutame valmis jagatud mudelit:
* **`alarjoeste/estonian-normalizer`**

Igal serveril piisab ainult mudeli allalaadimisest:
```bash
docker exec -it ollama ollama pull alarjoeste/estonian-normalizer
```

Kontroll:
```bash
docker exec -it ollama ollama list
```

Kiirtest:
```bash
docker exec -it ollama ollama run alarjoeste/estonian-normalizer "VĆ¤ljasta ainult JSON kujul {\"normalized_query\":\"...\"}. Normaliseeri pĆ¤ring: tere kas te saaksite Ć¶elda kes kinnitab hanke tulemuse"
```

NB!! `docker-compose.yml` tõmbab selle mudeli ka automaatselt `ollama` käivitumisel, seega pole vaja seda käsitsi laadida.

### Mudeli päritolu ja litsents
* Jagatud mudel: `alarjoeste/estonian-normalizer`
* Algallikas: EuroLLM-9B-Instruct.Q4_K_M.gguf
* Litsents: **Apache License 2.0**

Kursusetöö raportis tasub fikseerida kasutatud mudelinimi (`alarjoeste/estonian-normalizer`), et tulemused oleksid reprodutseeritavad.

