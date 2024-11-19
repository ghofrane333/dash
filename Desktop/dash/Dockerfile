FROM python:3.9-slim

WORKDIR /Desktop/dash

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["streamlit", "run", "dash.py"]


