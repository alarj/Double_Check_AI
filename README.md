# Double_Check_AI

tegemist on TalTech AI mikrokraadi kursusetööga

# Double_Check_AI: AI Guardrail Prototüüp

See projekt demonstreerib kahe-astmelist AI turvakihti (Guardrail), mis kasutab kahte erinevat mudelit, et tagada siseandmete turvalisus.

## Projekti struktuur

* **`logic/`** - Rakenduse tuumik.
    * `main.py` - Streamliti veebiliides ja AI loogika. Sisaldab automaatset mudelite laadimist ja protsessori (CPU) koormuse optimeerimist.
    * `Dockerfile` - Juhised Pythoni keskkonna ja sõltuvuste seadistamiseks.
    * `requirements.txt` - Vajalikud Pythoni paketid (Streamlit, Requests jne).
* **`docker-compose.yml`** - Orkestreerib kahte teenust: Ollama (AI mootor) ja Pythoni rakendus.
* **`.gitignore`** - Tagab, et mahukaid mudelifaile ja ajutisi faile ei lükata Giti hoidlasse.

## Kuidas käivitada

1.  **Klooni projekt:**
    ```bash
    git clone [https://github.com/alarj/Double_Check_AI.git](https://github.com/alarj/Double_Check_AI.git)
    cd Double_Check_AI
    git checkout main
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
4.  **Laadi alla AI mudelid (Kriitiline samm):**
    Esmakordsel seadistamisel tuleb mudelid käsitsi Ollama konteinerisse tõmmata, järgmistel käivitaitel pole seda enam vaja
    ```bash
    docker exec -it ollama ollama pull gemma2:2b
    docker exec -it ollama ollama pull llama3:8b
    ``` 
5.  **Ava rakendus:**
    Liides on kättesaadav aadressil `http://<serveri-ip>:8501` (serveri IP-aadressil pordis 8501).
    **NB!** serveri port 8501 peab olema internetist kättesaadav. Võib nõuda serveri või teenuspakkuja keskkonna eraldi häälestamist turvareeglite osas.

## Tehnoloogiad
* **Streamlit** - Veebiliides.
* **Ollama** - Lokaalne AI mudelite serveerimine.
* **Llama-3 (8B)** - Põhiline vastuste genereerimise mudel.
* **Gemma2 (2B)** - Kergekaaluline ja kiire kontrollmudel (Guardrail).
* **Phi-3 (mini)** - Kergekaaluline kontrollmoodul turvafiltri jaoks. (selle võib eraldi alla laadida kui huvi on, hetkel vaikimisi ei kasutata)

## Turvakihi loogika
Rakendus oskab kasutada kuni 3 astmelist protsessi:
1. **Eelkontroll**  analüüsib sisendit. Kui see on ohtlik, päringut põhimudelile ei saadeta.
2. **Põhipäring**  genereerib vastuse
3. **Järelkontroll** kontrollib väljundit, et vältida tundliku info leket või ebasobivat sisu.
