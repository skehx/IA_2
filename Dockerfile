FROM python:3.11-slim

WORKDIR /app/models

COPY models/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY models/ .

CMD ["python", "main.py"]
