# Étape 1 : Utiliser une image de base Python
FROM python:3.9-slim

# Étape 2 : Définir le répertoire de travail dans le conteneur
WORKDIR /dash

# Étape 3 : Copier les fichiers nécessaires dans le conteneur
COPY . /dash

# Étape 4 : Installer les dépendances à partir de requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Étape 5 : Exposer le port utilisé par l'application (par défaut 8000 pour Flask ou Streamlit)
EXPOSE 8000

# Étape 6 : Spécifier la commande à exécuter pour démarrer l'application
CMD ["gunicorn", "main:dash", "--bind", "0.0.0.0:8000"]
