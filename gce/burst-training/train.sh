#/bin/bash

# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

### Arguments
INSTANCE_NAME=$1
TRAINING_SCRIPT_DIR=$2
IMAGE_FAMILY=$3

TIMESTAMP=$(date +%s)

gsutil cp census-analysis.py "$TRAINING_SCRIPT_DIR/"

gcloud compute instances create \
  --machine-type=n1-standard-64 \
  --image-family=$IMAGE_FAMILY \
  --metadata-from-file startup-script=census-startup.sh \
  --metadata TRAINING_SCRIPT_DIR=$TRAINING_SCRIPT_DIR,TRAINING_SCRIPT_FILE=census-analysis.py,CENSUS_DATA_PATH=$TRAINING_SCRIPT_DIR/census,MODEL_OUTPUT_PATH=$TRAINING_SCRIPT_DIR/census-$TIMESTAMP.model,CV_ITERATIONS=300 \
  --scopes=cloud-platform \
  --preemptible \
  $INSTANCE_NAME
