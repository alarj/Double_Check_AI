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
    ```

2.  **Käivita Docker Compose:**
    ```bash
    docker compose up -d --build
    ```
    *Märkus: Esmakordsel käivitamisel laeb rakendus automaatselt alla Llama-3 (8B) ja Phi-3 (mini) mudelid. See võib võtta aega sõltuvalt internetikiirusest.*

3.  **Ava rakendus:**
    Liides on kättesaadav aadressil `http://localhost:8501` (või serveri IP-aadressil pordis 8501).

## Tehnoloogiad
* **Streamlit** - Veebiliides.
* **Ollama** - Lokaalne AI mudelite serveerimine.
* **Llama-3 (8B)** - Põhiline vastuste genereerimise mudel.
* **Phi-3 (mini)** - Kergekaaluline kontrollmoodul turvafiltri jaoks.
