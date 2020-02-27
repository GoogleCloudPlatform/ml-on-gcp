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


echo "NVIDIA Jasper Tutorial"
shopt -s expand_aliases

alias jasper_folder='cd ./DeepLearningExamples/PyTorch/SpeechRecognition/Jasper'

echo "Cloning NVIDIA DeepLearningExamples from github"
git clone https://github.com/NVIDIA/DeepLearningExamples.git

jasper_folder


echo "Building the Jasper PyTorch with TRT 6 container"
sh ./trt/scripts/docker/trt_build.sh


mkdir data checkpoint result

echo "Copying the notebook to the root directory of Jasper"
cp notebooks/JasperTRT.ipynb .


echo "Downloading pretrained Jasper model"
wget -nc -q --show-progress -O ./checkpoint/jasper_model.zip https://api.ngc.nvidia.com/v2/models/nvidia/jasperpyt_fp16/versions/1/zip
unzip -o ./checkpoint/jasper_model.zip -d ./checkpoint/


echo "Serving Jasper notebook"
jupyter notebook --ip=0.0.0.0 --allow-root JasperTRT.ipynb