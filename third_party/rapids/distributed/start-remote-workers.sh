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
# This script starts cuda in remote machines
trap "rm .f 2> /dev/null; exit" 0 1 3

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_ROOT/lib64
export NUMBAPRO_NVVM=$CUDA_ROOT/nvvm/lib64/libnvvm.so
export NUMBAPRO_LIBDEVICE=$CUDA_ROOT/nvvm/libdevice

export WORKERS_FILE="workers.txt"
export ZONE="us-central1-b"


while getopts ":cg" opt; do
  case ${opt} in
        c)
	    WORKER=dask-worker
            ;;
        g)
	    WORKER=dask-cuda-worker
            ;;
        \?) echo "Usage: cmd [-c] [-g]"
            exit
            ;;
        :)
            echo "Invalid option: $OPTARG requires an argument" 1>&2
            ;;
  esac
done
shift $((OPTIND -1))

function err() {
    # Handle error
    echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')]: $@" >&2
    exit 1
}

function gssh() {
    # Connect to remote GCP instance
    gcloud compute ssh $1 --internal-ip --zone ${ZONE} --command "$2"
}

function start_cuda() {
    # Connect to remote GCP instance and start cuda
    local SCHEDULER_IP=`hostname --all-ip-addresses | awk ' { print ( $1 )  } '`
    local SCHEDULER_PORT=8786

    for i in `cat ${WORKERS_FILE}`; do 
        echo "Starting ${WORKER} on node: $i"
        gssh $i "/home/jupyter/start-${WORKER}.sh $SCHEDULER_IP $SCHEDULER_PORT" &
    done
}

function main() {
    start_cuda || err "Unable to start CUDA in remote workers"
}

main

