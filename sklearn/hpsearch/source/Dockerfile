FROM python:2.7.13

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY worker.py ./

COPY gcs_helper.py ./
