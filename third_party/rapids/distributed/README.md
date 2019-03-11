# How to run Distributed RAPIDS


1. Make sure each node is using Deep Learning VM RAPIDS image.
2. Setup SSH access from master to worker nodes
3. Create file `workers.txt`
    Include hostnames for nodes in the cluster (include the master node as first line)
4. Copy the following files on master node, and each worker node in the cluster:

```bash
chmod +x start-remote-workers.sh
chmod +x start-dask-cuda-worker.sh
chmod +x start-dask-worker.sh
chmod +x stop-remote-workers.sh
```

5. On master node, start Dask scheduler:
  ```bash
  dask-scheduler &
  ```

6. On master node, launch script to start Dask CUDA on each worker node (this script will read workers.txt, so can start dask-cuda-worker on master node too)

```bash
  ./start-remote-workers.sh -g  # the ‘-g’ option is for GPUs (starts dask-cuda-worker). If want to test CPU-only use ‘-c’.
```

7. Run the sum.py Python script

```bash
cd <path> where sum.py resides
./run.sh -g 20 -d   # the “-d” option will use distributed dask (instead of local dask)
```

### Pre-requisites
 - Change the ZONE and REGION to the correct one below.
 - Modify `start-remote-workers.sh` with correct ZONE.
 - Modify `stop-remote-workers.sh` with correct ZONE.

### Instructions

Run the following commands from your Cloud Shell instance.

#### Define instance template

```bash
export PROJECT_NAME="[enter-your-project]"
export INSTANCE_TEMPLATE_NAME="rapids-distributed-template"
export IMAGE_FAMILY="rapids-latest-gpu-experimental"
export INSTANCE_GROUP_NAME="rapids-instance-group"
export ZONE="us-central1-b"
export REGION="us-central1"
export NUM_INSTANCES=5
export NUM_GPUS=4
export TOTAL_GPUS=$(($NUM_INSTANCES * $NUM_GPUS))
export STARTUP_SCRIPT="gsutil cp gs://cloud-samples-data/dlvm/rapids/rapids.zip /home/jupyter/rapids.zip && unzip /home/jupyter/rapids.zip -d /home/jupyter/ && chmod +x /home/jupyter/start-dask-worker.sh && chmod +x /home/jupyter/start-dask-cuda-worker.sh && chown -R jupyter:jupyter /home/jupyter/"


function create_instance_template() {
   # Create Instance Template.
   echo "Creating instance template"
    gcloud beta compute --project=${PROJECT_NAME} instance-templates create ${INSTANCE_TEMPLATE_NAME} \
         --machine-type=n1-standard-16 \
         --maintenance-policy=TERMINATE \
         --accelerator=type=nvidia-tesla-t4,count=${NUM_GPUS} \
         --min-cpu-platform=Intel\ Skylake \
         --tags=http-server,https-server \
         --image-family=${IMAGE_FAMILY} \
         --image-project=deeplearning-platform-release \
         --boot-disk-size=200GB \
         --boot-disk-device-name=${INSTANCE_TEMPLATE_NAME} \
         --scopes=https://www.googleapis.com/auth/cloud-platform \
         --metadata="install-nvidia-driver=True,proxy-mode=project_editors,startup-script=${STARTUP_SCRIPT}"
}
```

#### Create instance group

```bash
function create_instance_group() {
    # Create Virtual Machines.
    echo "Creating instance group"
    gcloud compute instance-groups managed create ${INSTANCE_GROUP_NAME} \
       --template ${INSTANCE_TEMPLATE_NAME} \
       --base-instance-name rapids-instances \
       --size ${NUM_INSTANCES} \
       --zones ${ZONE}
}
```

#### Define Project information

```bash
gcloud config set project ${PROJECT_NAME}
gcloud config set compute/zone ${ZONE}
gcloud config list

## Create instances

create_instance_template 
create_instance_group

## Create Firewall rule
gcloud compute firewall-rules create www-scheduler-8786 --target-tags http-server --allow tcp:8786
```

## Wait for instances to be created

Create `workers.txt` file

```bash
gcloud compute instance-groups list-instances ${INSTANCE_GROUP_NAME} --region ${REGION} | awk ' { print ( $1 )  } ' | tail -n +2 > workers.txt
MASTER=$(head -n 1 workers.txt)
```

Start dask-scheduler in Master node

```bash
gcloud compute scp ./workers.txt "jupyter@${MASTER}:/home/jupyter" --zone ${ZONE}
gcloud compute ssh "jupyter@${MASTER}" --zone ${ZONE} --command "nohup /opt/anaconda3/bin/dask-scheduler &" &
```
*Connection refused error means that either VM is not yet fully up or check firewall port 22



Start workers remotely and verify that cuda is working in each node

```bash
gcloud compute ssh "jupyter@${MASTER}" --zone ${ZONE} --command "cd /home/jupyter && nohup ./start-remote-workers.sh -g" &
```

Run job in a distributed way

```bash
gcloud compute ssh "jupyter@${MASTER}" --zone ${ZONE} --command "sudo chown -R jupyter:jupyter /home/jupyter && cd /home/jupyter/ && source /opt/anaconda3/bin/activate base && sudo chmod +x ./run.sh && ./run.sh -g ${TOTAL_GPUS} -d"
``` 

