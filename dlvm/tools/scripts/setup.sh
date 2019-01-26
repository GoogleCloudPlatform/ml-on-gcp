#!/bin/bash
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This script installs 2 virtual machines with auto-scaling.
trap "rm .f 2> /dev/null; exit" 0 1 3

export PROJECT_NAME=""
export INSTANCE_TEMPLATE_NAME="tf-inference-template"
export IMAGE_FAMILY="tf-1-12-cu100"
export INSTANCE_GROUP_NAME="deeplearning-instance-group"
export STARTUP_SCRIPT="gs://cloud-samples-data/dlvm/t4/start_agent_and_inf_server.sh"
export ZONES="us-central1-b"
export REGION="us-central1"
export NUM_INSTANCES=2
export UTILIZATION_TARGET=85

# Load Balancing parameters
export HEALTH_CHECK_NAME="http-basic-check"
export WEB_BACKED_SERVICE_NAME="tensorflow-backend"
export WEB_MAP_NAME="map-all"
export LB_NAME="tf-lb"
export IP4_NAME="lb-ip4"
export FORWARDING_RULE="lb-fwd-rule"


function err() {
  echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')]: $@" >&2
  exit 1
}

function check_exists() {
    if [ -z "$1" ]; then
          echo "$1 is not defined"
          return 1
    else
          echo ""
    fi
}

function create_instance_template() {
   # Create Instance Template.
   echo "Creating instance template"
    gcloud beta compute --project=${PROJECT_NAME} instance-templates create ${INSTANCE_TEMPLATE_NAME} \
         --machine-type=n1-standard-16 \
         --maintenance-policy=TERMINATE \
         --accelerator=type=nvidia-tesla-t4,count=4 \
         --min-cpu-platform=Intel\ Skylake \
         --tags=http-server,https-server \
         --image-family=${IMAGE_FAMILY} \
         --image-project=deeplearning-platform-release \
         --boot-disk-size=100GB \
         --boot-disk-type=pd-ssd \
         --boot-disk-device-name=${INSTANCE_TEMPLATE_NAME} \
         --metadata startup-script-url=${STARTUP_SCRIPT}
}

function create_instance_group() {
    # Create Virtual Machines.
    echo "Creating instance group"
    gcloud compute instance-groups managed create ${INSTANCE_GROUP_NAME} \
       --template ${INSTANCE_TEMPLATE_NAME} \
       --base-instance-name deeplearning-instances \
       --size ${NUM_INSTANCES} \
       --zones ${ZONES}
}

function verify_instances() {
  # Verify instances are created.
    gcloud compute instance-groups managed list-instances ${INSTANCE_GROUP_NAME} --region ${REGION}
    read -p "Waiting for instances to be created..." -t 120
    echo ""
    export INSTANCE_GROUP_NAME="deeplearning-instance-group"
    gcloud compute instance-groups managed list-instances ${INSTANCE_GROUP_NAME} --region ${REGION}
    # TODO(gogasca) Check instances are successfully created.
}


function enable_auto_scaling() {
    # Enable Auto-scaling
    gcloud compute instance-groups managed set-autoscaling ${INSTANCE_GROUP_NAME} \
       --custom-metric-utilization metric=custom.googleapis.com/gpu_utilization,utilization-target-type=GAUGE,utilization-target=${UTILIZATION_TARGET} \
       --max-num-replicas 4 \
       --cool-down-period 360 \
       --region ${REGION}
}

function create_load_balancing() {
    echo "Creating Load Balancing..."

    gcloud compute health-checks create http ${HEALTH_CHECK_NAME} \
       --request-path /v1/models/default \
       --port 8888

    gcloud compute instance-groups set-named-ports ${INSTANCE_GROUP_NAME} \
       --named-ports http:8888 \
       --region ${REGION}

    gcloud compute backend-services create ${WEB_BACKED_SERVICE_NAME} \
       --protocol HTTP \
       --health-checks ${HEALTH_CHECK_NAME} \
       --global

    gcloud compute backend-services add-backend ${WEB_BACKED_SERVICE_NAME} \
       --balancing-mode UTILIZATION \
       --max-utilization 0.8 \
       --capacity-scaler 1 \
       --instance-group ${INSTANCE_GROUP_NAME} \
       --instance-group-region ${REGION} \
       --global

    echo "Create URL maps"
    gcloud compute url-maps create ${WEB_MAP_NAME} \
       --default-service ${WEB_BACKED_SERVICE_NAME}

    echo "Create HTTP Proxy"
    gcloud compute target-http-proxies create ${LB_NAME} \
       --url-map ${WEB_MAP_NAME}

    # Add an external IP address to the load balancer
    echo "Adding an external IP address to the load balancer"
    gcloud compute addresses create ${IP4_NAME} \
       --ip-version=IPV4 \
       --global

    # Extract Public IP Address
    gcloud compute addresses list
    echo "Collecting Public IP address..."
    export IP=$(gcloud compute addresses list | grep ${IP4_NAME} | awk '{print $2}')

    echo "Setting up forwarding rules..."
    # Setup the forwarding rules
    gcloud compute forwarding-rules create ${FORWARDING_RULE} \
       --address ${IP} \
       --global \
       --target-http-proxy ${LB_NAME} \
       --ports 80
}

function enable_firewall() {
    echo "Enabling Firewall rules"
    # Enable Firewall
    gcloud compute firewall-rules create www-firewall-80 \
        --target-tags http-server --allow tcp:80

    gcloud compute firewall-rules create www-firewall-8888 \
        --target-tags http-server --allow tcp:8888
}


function firewall_status() {

    gcloud compute firewall-rules list

}

function delete_demo() {
    # Clean up all information in cluster
    echo "Deleting project setup..."
    echo
    echo "Deleting forwarding rules"
    gcloud -q compute forwarding-rules delete ${FORWARDING_RULE} --global

    echo "Deleting IPV4 address"
    gcloud -q compute addresses delete ${IP4_NAME} --global

    echo "Deleting load-balancer"
    gcloud -q compute target-http-proxies delete ${LB_NAME}

    echo "Deleting URL maps"
    gcloud -q compute url-maps delete ${WEB_MAP_NAME}

    echo "Deleting backend-service"
    gcloud -q compute backend-services delete ${WEB_BACKED_SERVICE_NAME} --global

    echo "Deleting health-checks"
    gcloud -q compute health-checks delete ${HEALTH_CHECK_NAME}

    echo "Deleting Instance group"
    gcloud -q compute instance-groups managed delete ${INSTANCE_GROUP_NAME} --region ${REGION}

    echo "Deleting Instance template"
    gcloud beta -q compute --project=${PROJECT_NAME} instance-templates delete ${INSTANCE_TEMPLATE_NAME}

    echo "Deleting Firewall rules"
    gcloud -q compute firewall-rules delete www-firewall-80
    gcloud -q compute firewall-rules delete www-firewall-8888

    echo "Delete has been completed"
}

function install_demo() {
    echo "┌─────────────────────────────────────────────┐"
    echo "│   Google Cloud Deep Learning VM NVIDIA T4   │"
    echo "└─────────────────────────────────────────────┘"
    echo " Demo installation starting..."
    check_exists ${PROJECT_NAME} || err "Project not defined"
    gcloud config set project ${PROJECT_NAME}
    gcloud config list
    # Deploy Virtual Machines + Load Balancer.
    create_instance_template || err "Unable to create instance template"
    create_instance_group || err "Unable to create instance groups"
    verify_instances || err "Instances not created"
    # Enable auto-scaling and load balancer.
    enable_auto_scaling || err "Unable to enable auto-scaling"
    create_load_balancing || err "Unable to enable load balancing"
    enable_firewall || err "Unable to enable firewall"
    read -p "Waiting for instances to boot..." -t 15
    echo "Cluster setup is completed!"
    echo
    echo "Verify Health check instances are healthy"

}

function usage() {
    echo "Usage $0 {cleanup..|install}"
    # provide more info about arguments for the start case
    # provide an example usage
}

function main() {
  # Select valid option.
  case "$1" in
    cleanup)
    echo "Do you wish to delete your demo cluster?"
      select yn in "Yes" "No"; do
        case $yn in
            # Install instances and load-balancer.
            Yes ) delete_demo; break;;
            No ) exit;;
        esac
      done;;
    install) install_demo;;
    enable_firewall) enable_firewall;;
    firewall_status) firewall_status;;
    *) usage ;;
  esac

}

main $1