FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends docker.io docker-cli docker-compose ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /bifrost

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8899"]
