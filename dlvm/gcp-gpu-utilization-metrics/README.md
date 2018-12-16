# Overview

This repository provides simple way to monitor GPU utilization on GCP

It is very simple to use, just run agent on each of your instance:

### Report metrics

This will auto create the metrics. 

```bash
git clone https://github.com/GoogleCloudPlatform/ml-on-gcp.git
cd dlvm/gcp-gpu-utilization-metrics
pip install -r ./requirements.txt
python report_gpu_metrics.py &
```

### Generate metrics

If you need to create metrics first run the following commands:

```bash
# Define your Google Cloud Project
GOOGLE_CLOUD_PROJECT=<PROJECT_ID>
git clone https://github.com/GoogleCloudPlatform/ml-on-gcp.git
cd dlvm/gcp-gpu-utilization-metrics
pip install -r ./requirements.txt
python create_gpu_metrics.py
```
