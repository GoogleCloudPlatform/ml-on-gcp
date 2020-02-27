#!/bin/bash

# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

shopt -s expand_aliases

alias data_folder='cd ./DeepLearningExamples/TensorFlow/LanguageModeling/BERT/data'
alias bert_folder='cd ..'


# NVIDIA-DOCKER is a prerequisite to launch this script
CONTAINER_VERSION='19.09-py3'

echo "Downloading NVIDIA container from NGC"
docker pull nvcr.io/nvidia/tensorflow:${CONTAINER_VERSION}

echo "Cloning NVIDIA DeepLearningExamples from github"
git clone https://github.com/NVIDIA/DeepLearningExamples.git

data_folder

echo "Downloading pre-trained models"
python3 -c 'from GooglePretrainedWeightDownloader import GooglePretrainedWeightDownloader
downloader = GooglePretrainedWeightDownloader("./download")
downloader.download()'

echo "Downloading SQuaD"
python3 -c 'from SquadDownloader import SquadDownloader
downloader = SquadDownloader("./download")
downloader.download()'

bert_folder

mv ./notebooks/bert_squad_tf_finetuning.ipynb .

echo "Running NGC container and Jupyter Lab"
nvidia-docker run \
  -v $PWD:/workspace/bert \
  -v $PWD/results:/results \
  --shm-size=1g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --publish 0.0.0.0:8888:8888 \
  -it  nvcr.io/nvidia/tensorflow:${CONTAINER_VERSION} /bin/sh -c 'cd /workspace/bert; jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser --NotebookApp.token="" --NotebookApp.password=""'

