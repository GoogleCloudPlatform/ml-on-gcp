
# Blog setup for Nvidia T4

## Overview

This script `ml-on-gcp/dlvm/tools/script/setup.sh` installs a demo cluster of 2 Compute instances
running TensorFlow serving.
Original setup from this [GCP blog](https://cloud.google.com/blog/products/ai-machine-learning/running-tensorflow-inference-workloads-at-scale-with-tensorrt-5-and-nvidia-t4-gpus)

### Instructions

1. Enable Cloud Compute Engine API in your Project.

2. Verify you have enough Nvidia GPU processors in your project. 

    - Products > IAM & admin > Quotas

3. Define your Project in setup.sh under ```export PROJECT_NAME=""```

Deploy instances:

```bash
setup.sh install
```
Delete demo:

```bash
setup.h cleanup
```

Enable firewall:

```bash
setup.h enable_firewall
```

Check firewall status:

```bash
setup.h firewall_status
```


### Testing prediction

```bash

curl -X POST $IP/v1/models/default:predict -d @/tmp/out.json
```

```bash
python inference.py
```

```bash
apt-get install apache2-utils -y
ab -n 30000 -c 150 -t 600 -g t4.tsv -H "Accept-Encoding: gzip,deflate" -p /tmp/out.json http://$IP/v1/models/default:predict
```

```bash
apt-get install gnuplot  
gnuplot
gnuplot> set terminal dumb
gnuplot> plot "out.data" using 9  w l
```


### Troubleshooting

Error:

```bash
Required 'compute.instanceTemplates.create' permission for 'projects/<Project name>/global/instanceTemplates/tf-inference-template'
```

Solution:

```bash
gcloud auth login
```

```bash
gcloud auth activate-service-account --key-file=<Your Key file>
```

Error:

```bash
NAME                         ZONE           STATUS  ACTION    LAST_ERROR
deeplearning-instances-6gck  us-central1-b          CREATING  Error QUOTA_EXCEEDED: Instance 'deeplearning-instances-6gck' creation failed: Quota 'NVIDIA_T4_GPUS' exceeded.  Limit: 1.0 in region us-central1.
deeplearning-instances-rz61  us-central1-b          CREATING  Error QUOTA_EXCEEDED: Instance 'deeplearning-instances-rz61' creation failed: Quota 'NVIDIA_T4_GPUS' exceeded.  Limit: 1.0 in region us-central1.
```

Solution:

Update your Quota to allow more than 1.0 NVIDIA T4