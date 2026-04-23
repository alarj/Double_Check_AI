# Double_Check_AI

tegemist on TalTech AI mikrokraadi kursusetĆ¶Ć¶ga

# Double_Check_AI: AI Guardrail PrototĆ¼Ć¼p

See projekt demonstreerib kahe-astmelist AI turvakihti (Guardrail), mis kasutab kahte erinevat mudelit, et tagada siseandmete turvalisus.

## Projekti struktuur

* **`logic/`** - Rakenduse tuumik.
    * `main.py` - Streamliti veebiliides ja AI loogika.
    * `Dockerfile` - Juhised Pythoni keskkonna ja sĆµltuvuste seadistamiseks.
    * `requirements.txt` - Vajalikud Pythoni paketid (Streamlit, Requests jne).
* **`data_pipeline/`** - Andmete ettevalmistus.
    * `ingest.py` - Skript XML failide tĆ¶Ć¶tlemiseks ja vektorbaasi saatmiseks.
    * `parsers.py` - Loogika konkreetsete hanke-XML struktuuride lugemiseks.
* **`storage/`** - (Kohalik andmehoidla, ignoreeritud Gitis)
    * `raw/procurements/` - Koht, kuhu kasutaja paneb oma XML failid.
    * `vector_db/` - ChromaDB poolt genereeritud vektorandmebaas.
* **`docker-compose.yml`** - Orkestreerib kahte teenust: Ollama (AI mootor) ja Pythoni rakendus.
* **`.gitignore`** - VĆ¤listab andmete, virtuaalkeskkonna ja mudelite sattumise Giti.

## Kuidas kĆ¤ivitada

1.  **Klooni projekt ja seadista keskkond:**
    ```bash
    git clone [https://github.com/alarj/Double_Check_AI.git](https://github.com/alarj/Double_Check_AI.git)
    cd Double_Check_AI
    git checkout main
    ```
	**Loo ja aktiveeri virtuaalkeskkond (andmete laadimiseks)**
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
    Anna oma kasutajale kohe ka Ćµigused, et dockerit ei peaks sudo kaudu iga kord kĆ¤ivitama
     ```bash
    sudo usermod -aG docker $USER
    ```
    Selleks, et ei peaks Ćµiguste muutmiseks vĆ¤lja logima
    ```bash
    newgrp docker
    ```
3.  **KĆ¤ivita Docker Compose:**
    ```bash
    docker compose up -d --build
    ```
    **NB!** Kui kĆ¤sk docker compose up lĆµpetab, kirjuta ```docker ps```, et nĆ¤ha, kas rakendus on "Up" staatuses ja millist porti ta kasutab!

4.  **Laadi alla AI mudelid (kriitiline samm):**
    Esmakordsel seadistamisel tuleb mudelid kĆ¤sitsi Ollama konteinerisse tĆµmmata, jĆ¤rgmistel kĆ¤ivitamistel pole seda enam vaja
	PĆµhimudelid:
    ```bash
    docker exec -it ollama ollama pull gemma2:2b
    docker exec -it ollama ollama pull llama3:8b
    ``` 
	Embedding mudel RAG jaoks:
	```bash
    docker exec -it ollama ollama pull mxbai-embed-large
    ``` 
	
5.  **Ava rakendus:**
    Liides on kĆ¤ttesaadav aadressil `http://<serveri-ip>:8501` (serveri IP-aadressil pordis 8501).
    **NB!** serveri port 8501 peab olema internetist kĆ¤ttesaadav. VĆµib nĆµuda serveri vĆµi teenuspakkuja keskkonna eraldi hĆ¤Ć¤lestamist turvareeglite osas.

## Andmete ettevalmistus (RAG)

SĆ¼steem kasutab RAG-loogikat (Retrieval-Augmented Generation). Enne rakenduse kasutamist tuleb andmed vektorbaasi laadida. Andmeid saab sisse logeda algandmetest, kui seda on vaja teha, siis tee nii:

### Eeltingimused
- Ollama peab olema installitud ja jooksma.

1.  **Kopeeri andmed**
    Pane oma hanke-XML failid kausta storage/raw/procurements/
	Pane seaduste failid storage/raw/laws/
	
	
2.  **KĆ¤ivita indekseerimine**
    *Hanked:*
	```bash
    source .venv/bin/activate
    cd data_pipeline
    python3 ingest.py
    ``` 
	*Seadused:*
	```bash
    source .venv/bin/activate
    cd data_pipeline
    python3 ingest_laws.py
    ``` 
	*Veebi juhendid:*
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
1. **Eelkontroll**  analĆ¼Ć¼sib sisendit. Kui see on ohtlik, pĆ¤ringut pĆµhimudelile ei saadeta.
2. **PĆµhipĆ¤ring**  genereerib vastuse
3. **JĆ¤relkontroll** kontrollib vĆ¤ljundit, et vĆ¤ltida tundliku info leket vĆµi ebasobivat sisu.

## Estonian normaliseerija mudel (tiimi serverid)
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

`docker-compose.yml` tĆµmbab selle mudeli ka automaatselt `ollama` kĆ¤ivitumisel.

### Mudeli pĆ¤ritolu ja litsents
* Jagatud mudel: `alarjoeste/estonian-normalizer`
* Algallikas: EuroLLM-9B-Instruct.Q4_K_M.gguf
* Litsents: **Apache License 2.0**

KursusetĆ¶Ć¶ raportis tasub fikseerida kasutatud mudelinimi (`alarjoeste/estonian-normalizer`), et tulemused oleksid reprodutseeritavad.

