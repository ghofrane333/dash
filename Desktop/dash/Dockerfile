FROM python:3.9-slim

# Définir le répertoire de travail
WORKDIR /Desktop/dash

# Copier les fichiers nécessaires dans l'image
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Exposer le port utilisé par Streamlit
EXPOSE 80

# Commande de démarrage
CMD ["streamlit", "run", "dash.py", "--server.port=80", "--server.enableCORS=false", "--server.address=0.0.0.0"]
