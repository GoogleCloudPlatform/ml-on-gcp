FROM python:2.7.13

COPY requirements.txt ./

# This is required for installing scikit-optimize
RUN pip install numpy==1.13.1

RUN pip install --no-cache-dir -r requirements.txt

COPY worker.py ./

COPY gcs_helper.py ./
