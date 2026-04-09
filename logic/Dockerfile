FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopeerime failid (sh main.py)
COPY . .

# --- LOGIFAILI ETTEVALMISTUS ---
# Tekitame tühja logifaili, et Docker ei peaks seda looma root-õigustega hiljem.
# chmod 666 lubab kõigil (nii konteineri-Pythonil kui Ubuntu-kasutajal) faili kirjutada.
RUN touch ai_turvakiht.log && chmod 666 ai_turvakiht.log

EXPOSE 8501

# Lisame lipud, mis lubavad ühendusi igalt poolt ja lülitavad kontrollid välja
CMD ["streamlit", "run", "main.py", \
    "--server.address=0.0.0.0", \
    "--server.port=8501", \
    "--server.enableCORS=false", \
    "--server.enableXsrfProtection=false", \
    "--server.headless=true"]
