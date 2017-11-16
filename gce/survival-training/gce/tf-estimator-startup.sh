#!/bin/bash

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


### Metadata specification
# All this metadata is pulled from the Compute Engine instance metadata server

# Your username on the GCE image
GCE_USER=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/gceUser -H "Metadata-Flavor: Google")

# HTTP URL to GitHub repository
TRAINER_GIT_PATH=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/trainerGitPath -H "Metadata-Flavor: Google")

# Module within repo that should be run (specify relative to repo root)
TRAINER_MODULE=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/trainerModule -H "Metadata-Flavor: Google")

# Directory in which training data lives (can be GCS path)
DATA_DIR=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/dataDir -H "Metadata-Flavor: Google")

# Directory in which checkpoinst should be stored (can be GCS path)
JOB_DIR=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/jobDir -H "Metadata-Flavor: Google")

# Trainer configuration
TRAIN_STEPS=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/trainSteps -H "Metadata-Flavor: Google")
NUM_GPUS=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/numGpus -H "Metadata-Flavor: Google")

MOMENTUM=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/momentum -H "Metadata-Flavor: Google")
WEIGHT_DECAY=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/weightDecay -H "Metadata-Flavor: Google")
LEARNING_RATE=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/learningRate -H "Metadata-Flavor: Google")
BATCH_NORM_DECAY=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/batchNormDecay -H "Metadata-Flavor: Google")
BATCH_NORM_EPSILON=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/batchNormEpsilon -H "Metadata-Flavor: Google")

# Set keepAlive=true in your instance metadata if you want the Compute Engine
# instance to stay running even after the training job is complete
KEEP_ALIVE=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/keepAlive -H "Metadata-Flavor: Google")


### Inject trainer into Compute Engine VM and set up the environment

cd "/home/$GCE_USER"

git clone $TRAINER_GIT_PATH trainer

cd "/home/$GCE_USER/trainer"

git pull origin master


### Run the job

sudo -u $GCE_USER python -m $TRAINER_MODULE \
  --data-dir=$DATA_DIR \
  --job-dir=$JOB_DIR \
  --train-steps=$TRAIN_STEPS \
  --num-gpus=$NUM_GPUS \
  --momentum=$MOMENTUM \
  --weight-decay=$WEIGHT_DECAY \
  --learning-rate=$LEARNING_RATE \
  --batch-norm-decay=$BATCH_NORM_DECAY \
  --batch-norm-epsilon=$BATCH_NORM_EPSILON


### Once the job has completed, unless $KEEP_ALIVE is true, shut down the
### Compute Engine instance

# Note that $KEEP_ALIVE is populated from the keepAlive metadata key.
if ! [ $KEEP_ALIVE = "true" ] ; then
  sudo shutdown -h now
fi
