#!/bin/bash


### Metadata specification
# All this metadata is pulled from the Compute Engine instance metadata server

# Your username on the GCE image
GCE_USER=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/gceUser -H "Metadata-Flavor: Google")

# Repo name (in Cloud Source Repositories)
TRAINER_REPO=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/trainerRepo -H "Metadata-Flavor: Google")

# Module within repo that should be run (specify relative to repo root)
TRAINER_MODULE=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/trainerModule -H "Metadata-Flavor: Google")

# Directory in which checkpoints should be stored
JOB_DIR=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/jobDir -H "Metadata-Flavor: Google")

# Trainer configuration
TRAIN_STEPS=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/trainSteps -H "Metadata-Flavor: Google")
CHECKPOINT_STEPS=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/checkpointSteps -H "Metadata-Flavor: Google")
HPARAM1=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/hparam1 -H "Metadata-Flavor: Google")
HPARAM2=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/hparam2 -H "Metadata-Flavor: Google")

# Set keepAlive=true in your instance metadata if you want the Compute Engine
# instance to stay running even after the training job is complete
KEEP_ALIVE=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/keepAlive -H "Metadata-Flavor: Google")


### Inject trainer into Compute Engine VM and set up the environment

cd "/home/$GCE_USER"

gcloud source repos clone "$TRAINER_REPO"

cd "/home/$GCE_USER/$TRAINER_REPO"

git pull origin master

mkdir -p $JOB_DIR


### Run the job

sudo -u $GCE_USER python -m $TRAINER_MODULE \
  --job-dir=$JOB_DIR \
  --train-steps=$TRAIN_STEPS \
  --checkpoint-steps=$CHECKPOINT_STEPS \
  --hyperparameter-1=$HPARAM1 \
  --hyperparameter-2=$HPARAM2


### Once the job has completed, unless $KEEP_ALIVE is true, shut down the
### Compute Engine instance

# Note that $KEEP_ALIVE is populated from the keepAlive metadata key.
if ! [ $KEEP_ALIVE = "true" ] ; then
  sudo shutdown -h now
fi
