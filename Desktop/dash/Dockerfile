
FROM python:3.9-slim

# Définir le répertoire de travail
WORKDIR /Desktop/dash

# Copier les fichiers nécessaires dans l'image
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste des fichiers de l'application
COPY . .

# Exposer le port attendu par Azure
EXPOSE 8000

# Commande de démarrage
CMD ["streamlit", "run", "dash.py", "--server.port=8000", "--server.enableCORS=false", "--server.address=0.0.0.0"]
