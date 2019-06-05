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
# This script installs GPU driver if needed. Run a notebook via Papermill. (https://github.com/nteract/papermill)

set -x -e

function err() {
  echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')]: $@" >&2
  exit 1
}


function install_gpu_driver() {
    # Verify if GPU driver are installed.
    if lspci -vnn | grep NVIDIA > /dev/null 2>&1; then
      # Nvidia card found, need to check if driver is up
      if ! nvidia-smi > /dev/null 2>&1; then
        echo "Installing driver"
        /opt/deeplearning/install-driver.sh
      fi
    fi
    return 0
}


function metadata_exists() {
    http_code=$(curl -i http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1 -H "Metadata-Flavor: Google" -o /dev/null -w '%{http_code}\n' -s)
    if [[ ${http_code} == 200 ]]; then
        return 0
    else
        return 1
    fi
}


function validate_metadata() {
    # Validates mandatory parameters exist.
    metadata_exists input_notebook_path || err "Input Notebook not defined"
    metadata_exists output_notebook_path || err "Output Notebook not defined"
    return 0
}


function run_notebook() {
  # Supports:
  #   papermill gs://bucket/notebook.ipynb gs://bucket/output/notebook.ipynb
  #   papermill gs://bucket/notebook.ipynb gs://bucket/output/notebook.ipynb -f gs://bucket/params.yaml
  #   papermill gs://bucket/notebook.ipynb gs://bucket/output/notebook.ipynb -p epochs 128
 
  # Add metadata attributes.
  INPUT_NOTEBOOK_PATH=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/input_notebook_path -H "Metadata-Flavor: Google")
  OUTPUT_NOTEBOOK_PATH=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/output_notebook_path -H "Metadata-Flavor: Google")
  TMP_NOTEBOOK_PATH='/tmp/notebook.ipynb'
  # Run Notebook using Papermill. https://github.com/nteract/papermill. Check if parameters option exists.
  metadata_exists parameters_file
  parameters_file_exists=$?
  if [[ ${parameters_file_exists} -eq 0 ]]; then
    # Passing parameters file
    echo "Parameters file exists, running notebook now..."
    PARAMETERS_FILE=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/parameters_file -H "Metadata-Flavor: Google")
    gsutil cp "${PARAMETERS_FILE}" params.yaml
    papermill "${INPUT_NOTEBOOK_PATH}" "${TMP_NOTEBOOK_PATH}" -f params.yaml --log-output    
  else
    metadata_exists parameters
    parameters_exists=$?
    if [[ ${parameters_exists} -eq 0 ]]; then        
      # Parameters as -p key value
      echo "Manual parameters defined, running notebook now..."
      PARAMETERS=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/parameters -H "Metadata-Flavor: Google")
      papermill "${INPUT_NOTEBOOK_PATH}" "${TMP_NOTEBOOK_PATH}" "${PARAMETERS}" --log-output
    else
      # No parameters
      echo "Running notebook now..."
      papermill "${INPUT_NOTEBOOK_PATH}" "${TMP_NOTEBOOK_PATH}" --log-output
    fi    
  fi
  # Copy file to avoid GCS limitation: https://github.com/nteract/papermill/issues/312
  gsutil cp "${TMP_NOTEBOOK_PATH}" "${OUTPUT_NOTEBOOK_PATH}"
}



function delete_instance(){
    # Delete Virtual Machine.
    INSTANCE_NAME=$(curl http://metadata.google.internal/computeMetadata/v1/instance/name -H "Metadata-Flavor: Google")
    INSTANCE_ZONE=$(curl http://metadata.google.internal/computeMetadata/v1/instance/zone -H "Metadata-Flavor: Google")
    INSTANCE_PROJECT_NAME=$(curl http://metadata.google.internal/computeMetadata/v1/project/project-id -H "Metadata-Flavor: Google")
    gcloud --quiet compute instances delete "${INSTANCE_NAME}" --zone "${INSTANCE_ZONE}" --project "${INSTANCE_PROJECT_NAME}"
}


function main() {
    install_gpu_driver || err "Installation of GPU driver failed"
    validate_metadata   || err "Invalid metadata"
    run_notebook || err "Processing notebook failed"
    metadata_exists stay_alive || delete_instance || err "Delete instance failed"
}

main
