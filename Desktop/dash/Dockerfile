# Utiliser une image Python comme base
FROM python:3.9-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers nécessaires
COPY . /app

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port (Azure l'utilise automatiquement)
EXPOSE 8000

# Commande pour démarrer l'application
CMD ["gunicorn", "dash:dash", "--bind", "0.0.0.0:8000"]

