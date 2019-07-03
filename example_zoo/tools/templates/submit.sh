# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


BUCKET=$EXAMPLE_ZOO_ARTIFACTS_BUCKET

TRAINER_PACKAGE_PATH="trainer"
MAIN_TRAINER_MODULE="trainer.{name}"

now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="{name}_$now"

JOB_DIR=$BUCKET"/"$JOB_NAME"/"

gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS

gcloud ai-platform jobs submit training $JOB_NAME \
    --job-dir $JOB_DIR  \
    --package-path $TRAINER_PACKAGE_PATH \
    --module-name $MAIN_TRAINER_MODULE \
    --region us-central1 \
    --config config.yaml \
    --runtime-version 1.13 \
    --python-version 2.7 \
    --project $EXAMPLE_ZOO_PROJECT_ID \
    -- \
    --model_dir=$JOB_DIR
