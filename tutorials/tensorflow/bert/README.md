# QA Inference on BERT using TensorRT

## The purpose of this docker:

  -  Create a TensorRT BERT Base(Large) Engine
  -  Run QA Inference on BERT Base(Large) by using the Engine previously created

## How to run this demo?

This demo can be using a simplified installation and launch the notebook from AI Hub and connect to the existing AI Platform notebook
or you can install the image from scratch.

### Simplified installation

#### Create Google Cloud Compute Engine instance with custom container

Create a Compute Engine with at least 100 GB Hard Disk

```bash
export CONTAINER="gcr.io/dpe-cloud-mle/tensorrt_bert_sample:staging"
export IMAGE_FAMILY="common-container"
export ZONE="us-central1-b"
export INSTANCE_NAME="bert-experiment"
export INSTANCE_TYPE="n1-standard-8"
gcloud compute instances create $INSTANCE_NAME \
        --zone=$ZONE \
        --image-family=$IMAGE_FAMILY \
        --image-project=deeplearning-platform-release \
        --machine-type=$INSTANCE_TYPE \
        --boot-disk-size=120GB \
        --maintenance-policy TERMINATE --restart-on-failure \
        --accelerator="type=nvidia-tesla-t4,count=1" \
        --scopes=https://www.googleapis.com/auth/cloud-platform \
        --metadata="install-nvidia-driver=True,proxy-mode=project_editors,container=${CONTAINER}" \
        --tags http-server,https-server       
```

Note:
   - You can create this instance in any available zone that supports T4 GPUs.
   - The option install-nvidia-driver=True installs NVIDIA GPU driver automatically.
   - The option proxy-mode=project_editors makes the VM visible in the [Notebook Instances section.](https://console.cloud.google.com/mlengine/notebooks/instances)
    

### Full installation

#### Create Google Cloud Compute Engine instance 

Create a Compute Engine with at least 100 GB Hard Disk

```bash
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

#### Login to instance

Login via SSH to the compute instance:

```bash

function gssh() {        
    gcloud compute ssh $1 --zone ${ZONE};  
}

gssh $INSTANCE_NAME
```

Stop Jupyter 

```
sudo systemctl stop jupyter.service
```

From terminal 

Copy Docker container

```
gsutil cp gs://aihub/assets/docker/tensorrt_bert_sample.tar.gz .
Copying gs://aihub/assets/docker/tensorrt_bert_sample.tar.gz...
/ [1 files][ 14.8 GiB/ 14.8 GiB]   61.1 MiB/s       
```

Unzip Docker container

```
gunzip tensorrt_bert_sample.tar.gz
ls -alh
total 20G
drwxr-xr-x  2 root root 4.0K Aug 12 04:13 .
drwxr-xr-x 23 root root 4.0K Aug 12 02:54 ..
-rw-r--r--  1 root root  20G Aug 12 02:57 tensorrt_bert_sample.tar
```

### 1. Run the docker image

Uncompress the tar.gz file and load it with docker (it will accept a tar file)

```
tensorrt_bert_sample.tar.gz
```

Load tar file.

```
docker load -i tensorrt_bert_sample.tar
```

This will create a new Docker image.


Rename image (Optional)

```
docker images
docker tag tensortrt_bert_demo:demo tensorrt_bert_sample:latest
```

Run Docker instances

```
nvidia-docker run  --publish 0.0.0.0:8080:8080 -e LD_LIBRARY_PATH=LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/tensorrt/lib -it tensorrt_bert_sample:latest bash
```


### 2. Test the TensorRT Engine creation

The starting directory is:

```
/workspace
```

Navigate to **/workspace/TensorRT/demo/BERT**

```
cd /workspace/TensorRT/demo/BERT/
```

From the docker container the folder structure is:

```
./
├── BERT_TRT.ipynb  (Jupyter Notebook to run inference)
├── build
│   ├── CMakeCache.txt
│   ├── CMakeFiles
│   ├── cmake_install.cmake
│   ├── libbert_plugins.so
│   ├── libcommon.so
│   ├── Makefile
│   └── sample_bert
├── CMakeLists.txt
├── data
│   ├── finetuned_model_fp32  (Contains the TF checkpoint for BERT Large)
│   │   ├── checkpoint
│   │   ├── model.ckpt-8144.data-00000-of-00001
│   │   ├── model.ckpt-8144.index
│   │   └── model.ckpt-8144.meta
│   ├── finetuned_model_base_fp32  (Contains the TF checkpoint for BERT Base)
│   │   ├── checkpoint
│   │   ├── model.ckpt-8144.data-00000-of-00001
│   │   ├── model.ckpt-8144.index
│   │   └── model.ckpt-8144.meta
│   ├── uncased_L-12_H-768_A-12 (BERT Base Configuration)
│   │   ├── bert_config.json
│   │   ├── bert_model.ckpt.data-00000-of-00001
│   │   ├── bert_model.ckpt.index
│   │   ├── bert_model.ckpt.meta
│   │   └── vocab.txt
│   └── uncased_L-24_H-1024_A-16  (BERT Large Configuration)
│       ├── bert_config.json
│       ├── bert_model.ckpt.data-00000-of-00001
│       ├── bert_model.ckpt.index
│       ├── bert_model.ckpt.meta
│       └── vocab.txt
│  
├── data_processing.py
├── python
│   ├── convert_records.py
│   ├── convert_weights.py
│   └── generate_dbg.py
├── README.md
├── sampleBERT.cpp
├── ...
└── tokenization.py
```

The script to launch is **bert_model.py**

```
usage: bert_model.py [-h] -m MODEL -o OUTPUT [-b BATCHSIZE] [-s SEQUENCE] -c
                     CONFIG
```
Typical configuration is (for BERT Large):
```
python bert_model.py -m "./data/finetuned_model_fp32/model.ckpt-8144" -o ./bert_python.engine -c ./data/uncased_L-24_H-1024_A-16/
```

For BERT Base (This is our configuration of choice)
```
python bert_model.py -m "./data/finetuned_model_base_fp32/model.ckpt-8144" -o ./bert_python_base.engine -c ./data/uncased_L-12_H-768_A-12/
```

Verify you see:
```
Serializing the engine....
Saving the engine....
Done.
```

_If this does not work it is possible that you need to rebuild the c++ binary sampleBERT.cpp_.
_See Paragraph 4 for this_

 ## 3. Running Inference with the Jupyter Notebook

 From the working directory:
 ```
 /workspace/TensorRT/demo/BERT
 ```

Run:

```
pip install -U jupyterlab
jupyter lab --ip=0.0.0.0 --port=8080 --allow-root --no-browser --NotebookApp.token='' --NotebookApp.password=''
```

Now you should be able to connect with a browser and run the notebook:

```
BERT_TRT.ipynb
```

Note: Verify you open the firewall, when accessing the notebook for external. Open TCP port 8080.

The default configuration if set to run BERT base but there are options
to also run BERT Large un-commenting the lines for the TensorRT engine location
and the vocab.txt file.


# 4. Compile TensorRT Demo BERT

It may be possible that the TensorRT plugins need to be re-built
in order to do so please follow the steps below:

Ref: [Github: TensortRT/demo/BERT](https://github.com/NVIDIA/TensorRT/tree/release/5.1/demo/BERT)

Please follow the steps described in the README.md file of the link above
(the source files are already included in the docker, so you don't need to download anything).
Use the docker container included to build the binary of sampleBERT.cpp.

The sample will compare outputs from TensorFlow and TRT BERT models given the same inputs,
but you don't need to run it.
Once it compiles you may want to go back to point 2