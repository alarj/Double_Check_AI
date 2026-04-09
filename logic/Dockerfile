FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501

# Lisame lipud, mis lubavad ühendusi igalt poolt ja lülitavad kontrollid välja
CMD ["streamlit", "run", "main.py", \
    "--server.address=0.0.0.0", \
    "--server.port=8501", \
    "--server.enableCORS=false", \
    "--server.enableXsrfProtection=false", \
    "--server.headless=true"]
