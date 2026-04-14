# Kasutame kergemat Pythoni baaspilti
FROM python:3.10-slim

# Määrame töökausta konteineris
WORKDIR /app

# Paigaldame süsteemsed sõltuvused, mis on vajalikud võrgusuhtluseks
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Kopeerime requirements faili ja paigaldame Pythoni paketid
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopeerime kogu loogika kausta sisu konteinerisse
# (See on vajalik, et API ja UI kood oleks olemas)
COPY . .

# Tekitame tühja logifaili ja anname sellele kirjutamisõigused
RUN touch /app/ai_turvakiht.log /app/api_access.log && chmod 666 /app/ai_turvakiht.log /app/api_access.log

# Portide avamine (dokumentatsiooniks)
EXPOSE 8501
EXPOSE 8000

# Dockerfile ise ei määra lõplikku käsku, kuna compose failis on 
# 'logic-app' ja 'api' jaoks erinevad 'command' väärtused.
