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

GCE_USER=$GCE_USER
DATA_DIR=$DATA_DIR
JOB_DIR=$JOB_DIR

TRAINER_GIT_PATH="https://github.com/tensorflow/models.git"
TRAINER_MODULE="tutorials.image.cifar10_estimator.cifar10_main"

TRAIN_STEPS=99999999
NUM_GPUS=4

# Hyperparameters
MOMENTUM=0.9
WEIGHT_DECAY=0.0002
LEARNING_RATE=0.1
BATCH_NORM_DECAY=0.997
BATCH_NORM_EPSILON=0.00001


### Push metadata to GCE instance

gcloud compute instances add-metadata cifar10-estimator --zone us-west1-b \
    --metadata-from-file startup-script=./gce/tf-estimator-startup.sh \
    --metadata "gceUser=$GCE_USER,trainerGitPath=$TRAINER_GIT_PATH,trainerModule=$TRAINER_MODULE,dataDir=$DATA_DIR,jobDir=$JOB_DIR,trainSteps=$TRAIN_STEPS,numGpus=$NUM_GPUS,momentum=$MOMENTUM,weightDecay=$WEIGHT_DECAY,learningRate=$LEARNING_RATE,batchNormDecay=$BATCH_NORM_DECAY,batchNormEpsilon=$BATCH_NORM_EPSILON"
