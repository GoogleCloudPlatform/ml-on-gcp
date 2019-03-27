# Overview

This repository provides simple way to monitor GPU utilization on GCP

It is very simple to use, just run agent on each of your instance:

### Report metrics

Installs a monitoring agent that monitors the GPU usage on the instance.
This will auto create the GPU metrics.

```bash
git clone https://github.com/GoogleCloudPlatform/ml-on-gcp.git
cd ml-on-gcp/dlvm/gcp-gpu-utilization-metrics 
pip install -r ./requirements.txt
python report_gpu_metrics.py &
```

### Generate metrics

If you need to create metrics using create_metric_descriptor first run the following commands:

```bash
# Define your Google Cloud Project
export GOOGLE_CLOUD_PROJECT=<PROJECT_ID>
git clone https://github.com/GoogleCloudPlatform/ml-on-gcp.git
cd dlvm/gcp-gpu-utilization-metrics
pip install -r ./requirements.txt
python create_gpu_metrics.py
```
Example:

```
Created projects/project-sample/metricDescriptors/custom.googleapis.com/gpu_utilization.
Created projects/project-sample/metricDescriptors/custom.googleapis.com/gpu_memory_utilization.
```

### Troubleshooting


Problem: Error when running report_gpu_metrics

```
google.api_core.exceptions.InvalidArgument: 400 One or more TimeSeries could not be written: One or more points were written more frequently than the maximum sampling period configured for the metric.
: timeSeries[0]
```

Solution:
Verify a single instance of this process is running in background

```
ps aux | grep "[r]eport_gpu_metrics.py"
```

Problem: Warning when running report_gpu_metrics or create_gpu_metrics
```
/usr/local/src/venv/slava/lib/python3.5/site-packages/google/auth/_default.py:66: UserWarning: Your application ha
s authenticated using end user credentials from Google Cloud SDK. We recommend that most server applications use service accounts instead. If your application continues to use end user credentials fro
m Cloud SDK, you might receive a "quota exceeded" or "API not enabled" error. For more information about service accounts, see https://cloud.google.com/docs/authentication/
  warnings.warn(_CLOUD_SDK_CREDENTIALS_WARNING)
```

Solution:
Configure GOOGLE_APPLICATION_CREDENTIALS using a service account.

```
export GOOGLE_APLICATION_CREDENTIALS=/usr/local/src/credentials.json
```
