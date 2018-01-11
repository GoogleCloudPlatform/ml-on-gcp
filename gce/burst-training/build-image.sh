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

#!/bin/bash

IMAGE_INSTANCE_NAME=$1
IMAGE_FAMILY=$2
IMAGE_NAME="${IMAGE_FAMILY}-$(date +%s)"

gcloud compute instances create --image-family $IMAGE_FAMILY --metadata-from-file startup-script=image.sh $IMAGE_INSTANCE_NAME

while :
do
  sleep 30
  echo "Polling for status of $IMAGE_INSTANCE_NAME ..."
  STATUS=$(gcloud compute instances list --filter $IMAGE_INSTANCE_NAME --format 'value(STATUS)')
  echo "Status: $STATUS"
  if [[ $STATUS == "TERMINATED" ]]; then
        break
  fi
done

gcloud compute images create --source-disk $IMAGE_INSTANCE_NAME --family $IMAGE_FAMILY $IMAGE_NAME

gcloud compute instances delete $IMAGE_INSTANCE_NAME --quiet

