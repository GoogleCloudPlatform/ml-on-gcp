#!/bin/bash

TRAINER_REPO=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/trainerRepo -H "Metadata-Flavor: Google")
TRAINER_MODULE=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/trainerModule -H "Metadata-Flavor: Google")
JOB_DIR=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/jobDir -H "Metadata-Flavor: Google")
TRAIN_STEPS=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/trainSteps -H "Metadata-Flavor: Google")
CHECKPOINT_STEPS=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/checkpointSteps -H "Metadata-Flavor: Google")
HPARAM1=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/hparam1 -H "Metadata-Flavor: Google")
HPARAM2=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/hparam2 -H "Metadata-Flavor: Google")

gcloud source repos clone "$TRAINER_REPO"

cd "$TRAINER_REPO"

git pull origin master

python -m "$TRAINER_MODULE" \
  --job-dir="$JOB_DIR" \
  --train-steps="$TRAIN_STEPS" \
  --checkpoint-steps="$CHECKPOINT_STEPS" \
  --hyperparameter-1="$HPARAM1" \
  --hyperparameter-2="$HPARAM2"

sudo shutdown -h now
