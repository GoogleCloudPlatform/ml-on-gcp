# Overview

This repository provides simple way to monitor GPU utilization on GCP

It is very simple to use, just run agent on each of your instance:

```bash
git clone https://github.com/b0noI/gcp-gpu-utilization-metrics.git
cd gcp-gpu-utilization-metrics
pip install -r ./requirements.txt
python ./report_gpu_metrics.py &
```

This will auto create the metrics. But if you need to create metrics first run the following commands:

```bash
git clone https://github.com/b0noI/gcp-gpu-utilization-metrics.git
cd gcp-gpu-utilization-metrics
pip install -r ./requirements.txt
GOOGLE_CLOUD_PROJECT=<ID> python ./create_gpu_metrics.py
```
