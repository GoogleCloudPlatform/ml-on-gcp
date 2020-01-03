```
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
```
<img src="http://developer.download.nvidia.com/compute/machine-learning/frameworks/nvidia_logo.png" style="width: 90px; float: right;">


# Jasper Inference For TensorRT


## Create Google Cloud Compute Engine instance

Create a Compute Engine with at least 100 GB Hard Disk

```
export IMAGE_FAMILY="common-cu101"
export ZONE="us-central1-b"
export INSTANCE_NAME="bert-experiment"
export INSTANCE_TYPE="n1-standard-16"
gcloud compute instances create $INSTANCE_NAME \
        --zone=$ZONE \
        --image-family=$IMAGE_FAMILY \
        --image-project=deeplearning-platform-release \
        --machine-type=$INSTANCE_TYPE \
        --boot-disk-size=120GB \
        --maintenance-policy TERMINATE --restart-on-failure \
        --accelerator="type=nvidia-tesla-t4,count=2" \
        --scopes=https://www.googleapis.com/auth/cloud-platform \
        --metadata="install-nvidia-driver=True,proxy-mode=project_editors" \
        --tags http-server,https-server
```

## Login to instance

Login via SSH to the compute instance:

```
function gssh() {
    gcloud compute ssh $1 --zone ${ZONE};
}

gssh $INSTANCE_NAME
```

## Stop Jupyter

```
sudo systemctl stop jupyter.service
```

## From terminal

Copy  script `build_asset.sh`

```
gsutil cp gs://aihub/assets/script/build_asset.sh .

```

## Run Script

Run the script to:
 - download NGC container,
 - clone the DeepLearningExample repository from github
 - download the data and the pre-trained models
 - launch jupyter lab

```
sh build_asset.sh
```


_Note: Verify you open the firewall, when accessing the notebook for external. Open TCP port 8888._
