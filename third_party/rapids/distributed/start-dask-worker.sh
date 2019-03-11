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
# This script starts Dask worker.
trap "rm .f 2> /dev/null; exit" 0 1 3

source /opt/anaconda3/bin/activate base

export CUDA_ROOT=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_ROOT/lib64
export NUMBAPRO_NVVM=$CUDA_ROOT/nvvm/lib64/libnvvm.so
export NUMBAPRO_LIBDEVICE=$CUDA_ROOT/nvvm/libdevice

function err() {
    # Handle error
    echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')]: $@" >&2
    exit 1
}

function start_dask() {
	local SCHEDULER_IPADDR=$1
	local SCHEDULER_PORT=$2
	/opt/anaconda3/bin/dask-worker ${SCHEDULER_IPADDR}:${SCHEDULER_PORT}		
}

function main() {
    start_dask $1 $2 || err "Unable to start CUDA worker"
}

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Improper arguments supplied..."
    echo "Exiting..."
    exit
fi

main $1 $2