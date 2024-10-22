# Overview

This repository provides simple way to monitor GPU utilization on GCP.
It supports the following NVIDIA Accelerators:
 -  A100
 -  H100
 -  L4
 -  K80
 -  P100
 -  P4
 -  V100
 -  T4

It is very simple to use, just run the agent on each of your Compute
Engine instances:

### Create metrics

Creates the metrics in Google Cloud StackDriver

```
git clone https://github.com/GoogleCloudPlatform/ml-on-gcp.git
cd ml-on-gcp/dlvm/gcp-gpu-utilization-metrics 
pip install -r ./requirements.txt
python create_gpu_metrics.py 
```

Example:

```
Created projects/project-sample/metricDescriptors/custom.googleapis.com/utilization_memory.
Created projects/project-sample/metricDescriptors/custom.googleapis.com/utilization_gpu.
Created projects/project-sample/metricDescriptors/custom.googleapis.com/memory_used
```

### Report metrics

Installs a monitoring agent that monitors the GPU usage on the instance.

```bash
git clone https://github.com/GoogleCloudPlatform/ml-on-gcp.git
cd ml-on-gcp/dlvm/gcp-gpu-utilization-metrics 
pip install -r ./requirements.txt
python report_gpu_metrics.py &
```

### Adding more metrics

If you want to add more metrics, just do the following changes:

1. Edit [create_gpu_metrics.py](create_gpu_metrics.py) and add a new
   method for each metric you need. Example: I want to add `memory.free`
   parameter.
    
   ```python
   add_new_metric(project_id, 'memory_free', 'Metric for amount of GPU
   memory used.')
   ```

2. Edit [report_gpu_metrics.py](report_gpu_metrics.py) and add a new method under
   `report_metrics` Example:
   
   ```
   report_metric(
            value=get_metric_value(metrics.get('memory_free')),
            metric_type='memory_free',
            resource_values=resource_values)
   ```

Currently we support the parameters defined
[here](https://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries)
Feel free to contribute and add more parameters to `GpuMetrics` in 
[report_gpu_metrics.py](report_gpu_metrics.py)

## Generate GPU service.

```bash 
cat <<-EOH > /lib/systemd/system/gpu_utilization_agent.service
[Unit]
Description=GPU Utilization Metric Agent
[Service]
Type=simple
PIDFile=/run/gpu_agent.pid
ExecStart=/bin/bash --login -c '/usr/bin/python /root/report_gpu_metrics.py'
User=root
Group=root
WorkingDirectory=/
Restart=always
[Install]
WantedBy=multi-user.target
EOH
```
# Reload systemd manager configuration

```
systemctl daemon-reload
```

# Enable gpu_utilization_agent service

```
systemctl --no-reload --now enable /lib/systemd/system/gpu_utilization_agent.service
```

### Testing

Use the gpu-burn tool to load your GPU to 100% utilization for 600 seconds:

```
git clone https://github.com/GoogleCloudPlatform/ml-on-gcp.git
cd ml-on-gcp/third_party/gpu-burn
make
./gpu_burn 600 > /dev/null &
```

### Troubleshooting


Problem: Error when running [report_gpu_metrics.py](report_gpu_metrics.py)

```
google.api_core.exceptions.InvalidArgument: 400 One or more TimeSeries could not be written: One or more points were written more frequently than the maximum sampling period configured for the metric.
: timeSeries[0]
```

Solution:
Verify a single instance of this process is running in background

```
ps aux | grep "[r]eport_gpu_metrics.py"
```

Problem: Warning when running
[report_gpu_metrics.py](report_gpu_metrics.py) or
[create_gpu_metrics.py](create_gpu_metrics.py)
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

Problem: 404 Error when running
[report_gpu_metrics.py](report_gpu_metrics.py)

```
 'Failed to retrieve
http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/?recursive=true
```

Solution: Your Compute Engine needs Google API Cloud access. Allow Read 
access in "Access scopes" Compute Engine, Stackdriver Logging and
Monitoring.
