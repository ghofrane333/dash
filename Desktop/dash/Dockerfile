FROM python:3.9-slim

WORKDIR /Desktop/dash

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Exposer le port que votre application écoute
EXPOSE 8000

CMD ["streamlit", "run", "dash.py", "--server.port=8000", "--server.address=0.0.0.0"]


