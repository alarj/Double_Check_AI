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
    **NB!** Kuna kogu rakendus töötab dockeri konteinerites, siis selle venv osa võib vahele jätta, seda pole reaalselt vaja.
	Kõiki käske ja teste on mõistlik jooksutada dockeris
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
**NB! Vektorbaasi binaarfaile ei hoita enam Gitis.** Tiim kasutab jagatud snapshot'i (zip-fail), et kõik testiksid sama andmestikuga.
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

## Vektorbaasi jagamine tiimiga

Tiimi sees jagame Chroma vektorbaasi **zipitud snapshot'ina**, mitte Git commitite kaudu. See väldib suuri binaarfaile repo ajaloos ja tagab, et kõik töötavad sama andmestiku peal.

### Soovitatud töövoog

1. Üks tiimiliige ehitab vektorbaasi valmis.
2. Kaust `storage/vector_db/` pakitakse zip-faili.
3. Zip laaditakse üles **GitHub Release assetina**.
4. Teised tiimiliikmed tõmbavad selle serverisse `wget` abil.
5. Vana vektorbaas kustutatakse enne taastamist ära.
6. Zip pakitakse lahti kausta `storage/`.
7. Teenused taaskäivitatakse.

### Snapshot'i loomine serveris

#### 1. Veendu, et vektorbaas on valmis

Kontrollnäide:
```bash
docker exec -it logic-app python -c "import chromadb; client = chromadb.PersistentClient(path='/app/storage/vector_db'); c = client.get_collection('procurements'); print('count:', c.count())"
```

#### 2. Paki `storage/vector_db/` zip-failiks

```bash
cd ~/Double_Check_AI/storage
zip -r vector_db_rhs_vos_bge-m3_2026-04-29.zip vector_db
```

Soovituslik nimetus:
```text
vector_db_<andmestik>_<embedding-mudel>_<kuupäev>.zip
```

Näiteks:
```text
vector_db_rhs_vos_bge-m3_2026-04-29.zip
```

#### 3. Laadi zip GitHub Release'i üles

Kasuta olemasolevat release'i või tee uus release. GitHubi veebis:
1. Ava repo `Releases`
2. Vali `Draft a new release` või ava olemasolev release
3. Lisa asset:
   * `vector_db_rhs_vos_bge-m3_2026-04-29.zip`
4. Lisa kirjeldusse vähemalt:
   * sisaldab: `RHS`, `VÕS`
   * embedding-mudel: `bge-m3`
   * loomise kuupäev
   * kasutatud ingest skript: `data_pipeline/ingest_laws.py`

### Snapshot'i taastamine teises serveris

#### 1. Mine projekti `storage/` kausta

```bash
cd ~/Double_Check_AI/storage
```

#### 2. Kustuta vana vektorbaas

Jäta soovi korral `.gitkeep` alles:
```bash
find vector_db -mindepth 1 ! -name '.gitkeep' -exec rm -rf {} +
```

Pärast kontroll:
```bash
ls -la ~/Double_Check_AI/storage/vector_db
```

#### 3. Tõmba uus zip GitHub Release'ist serverisse

Kopeeri GitHub Release asseti otselink ja kasuta `wget`-i:

```bash
wget "GITHUB_RELEASE_ASSET_URL" -O vector_db_rhs_vos_bge-m3_2026-04-29.zip
```

Näiteks:
```bash
wget "https://github.com/alarj/Double_Check_AI/releases/download/v3.0/vector_db_rhs_vos_bge-m3_2026-04-29.zip" -O vector_db_rhs_vos_bge-m3_2026-04-29.zip
```

#### 4. Paki zip lahti

```bash
unzip vector_db_rhs_vos_bge-m3_2026-04-29.zip
```

Kui zip sisaldab sees kausta `vector_db/`, siis taastub õige struktuur automaatselt.

#### 5. Kontrolli, et failid on olemas

```bash
ls -la ~/Double_Check_AI/storage/vector_db
```

Oodatav tulemus:
* `chroma.sqlite3`
* vähemalt üks UUID-nimeline alamkaust
* võimalik `.gitkeep`

#### 6. Taaskäivita teenused

```bash
cd ~/Double_Check_AI
docker compose restart api logic-app
```

#### 7. Kontrolli, et vektorbaas on loetav

```bash
docker exec -it logic-app python -c "import chromadb; client = chromadb.PersistentClient(path='/app/storage/vector_db'); c = client.get_collection('procurements'); print('count:', c.count())"
```

### Kas algfaile on vaja?

Süsteemi tavapäraseks tööks **ei ole** seaduste algfaile vaja, kui valmis vektorbaasi snapshot on juba taastatud kausta `storage/vector_db/`.

See tähendab:
* retrieval ja vastamine töötavad ainult vektorbaasi põhjal
* `storage/raw/laws/` algfaile ei ole vaja lihtsalt süsteemi kasutamiseks

Algfaile on vaja ainult siis, kui soovid:
* vektorbaasi uuesti nullist ehitada
* lisada uusi seadusfaile
* muuta ingest loogikat ja andmed uuesti sisse lugeda
* kontrollida, millest konkreetne chunk tekkis

### Millal snapshot uuendada

Tee uus snapshot siis, kui:
* lisatakse uus seadusfail
* muudetakse ingest loogikat
* vahetatakse embedding-mudelit
* tehakse täielik vektorbaasi rebuild

Kui snapshot muutub, uuenda ka selle nimi ja GitHub Release'i kirjeldus, et tiim teaks, milline versioon on kasutusel.

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

Post-check reeglimudeli tehniline spetsifikatsioon asub failis:
* **`docs/post_check_reeglimudel.md`**

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

## Gemini normaliseerija (väline API)
`/normalize` endpoint oskab lisaks Ollama mudelitele kasutada ka Google Gemini API teenust.

Kasutamine:
1. Sea API võti keskkonnamuutujasse:
```bash
export GEMINI_API_KEY="<sinu_gemini_api_voti>"
```
2. UI-s vali normaliseerimismudeliks näiteks:
* `gemini:gemini-2.5-flash`

Vorming:
* Kui mudelinimi algab `gemini:`, kasutab API Google Gemini teenust.
* Muul juhul kasutab API Ollama mudelit.

Näidis `/normalize` request:
```json
{
  "user_input": "kui suur on lihthanke piirmäär",
  "model": "gemini:gemini-2.5-flash",
  "timeout": 90,
  "threads": 4
}
```

## Automaattestid
Süsteemis on olemas järgmised testid, mis asuvad /testing/ kataloogis:

* **bench-pre-check.py** -- pre-check test, testilood failis pre_check_dataset.json
* **bench-post-check.py** -- post-check test, testilood failis post_check_dataset.json
* **test_post_check_use_cases.py** -- post-check kasutusjuhtude test (eraldi 4a sisuline + 4b turvakontroll, soovi korral ka koondkontroll), testilood failis post_check_use_cases_dataset.json ja testi tööparameetrid failis tests_conf.json (`tests.post-check-use-cases`)
* **retr-test.py** -- retrieval (vektorbaasi päring) test, testilood failis retrieval_dataset.json
* **llm-test.py** -- põhipäringu test, testilood failis main_llm_dataset.json
* **stability-test.py** -- kordab sama RAG-päringut ning kontrollib, kas retrieval'i kontekst ja selle konteksti põhjal antud põhipäringu vastus püsivad samad. Pre-check, normaliseerimine ja post-check ei ole selle testi osa. Vaikimisi küsimus, korduste arv, threadide arv, `timeout`, `n_results`, `max_context_blocks` ja `pause_seconds` tulevad failist `/testing/tests_conf.json`; käsurea parameetriga saab neid jooksu ajaks üle kirjutada.
* **normalizer-test.py** -- normaliseerimise test, testilood failis normalizer_dataset.json
* **benchmark_embeddings.py** -- testib erinevaid embeddingu mudeleid, st **NB!** koostab erinevate mudelitega vektorandmebaasid ja võrdleb neid omavahel. Veidi ajas maha jäänud. 

Kõik testid koostavad ka logi nii ekraanile kui .json formaadis sinnasamasse /testing/ kataloogi. Logid on nähtavad ka UI kaudu ja API teenuses /logs Mõlemast saab neid alla laadida ja oma arvutis säilitada või edasi uurida.
**NB!** Hetkel serveri käsureal kuvatav info ei kajastu logifailis -- see tuleb eraldi ära salvestada. Selle lisamine logisse on üks ToDo task ;-)

* Testid käivitatakse serveri käsurealt 
```bash
docker exec -it test-app python /testing/testifaili_nimi.py
```
(testifaili_nimi.py asemele tuleb siis kirjutada õige faili nimi vastavalt eespool olevale infole)

Stabiilsuse testi näited:
```bash
docker exec -it test-app python /testing/stability-test.py --case-index 1 --repeat 100
docker exec -it test-app python /testing/stability-test.py --question "kes kinnitab hanke tulemuse?" --repeat 1000
```

 

