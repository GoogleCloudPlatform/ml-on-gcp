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
# Install NVIDIA driver and create GPU utilization service.

# Install NVIDIA driver
sh /opt/deeplearning/install-driver.sh

# GPU Agent
git clone https://github.com/GoogleCloudPlatform/ml-on-gcp.git
cd ml-on-gcp/dlvm/gcp-gpu-utilization-metrics
# Install Python dependencies.
pip install -r ./requirements.txt
cp ./report_gpu_metrics.py /root/report_gpu_metrics.py

# Generate GPU service.
cat <<-EOH > /lib/systemd/system/gpu_utilization_agent.service
[Unit]
Description=GPU Utilization Metric Agent
[Service]
Type=simple
PIDFile=/run/gpu_agent.pid
ExecStart=/bin/bash --login -c '/usr/bin/python /root/report_gpu_metrics.py'
User=root
Group=root
WorkingDirectory=/
Restart=always
[Install]
WantedBy=multi-user.target
EOH
# Reload systemd manager configuration
systemctl daemon-reload
# Enable gpu_utilization_agent service
systemctl --no-reload --now enable /lib/systemd/system/gpu_utilization_agent.service

# Generate TF Service service.
cat <<-EOH > /lib/systemd/system/tfserve.service
[Unit]
Description=Inf Logic
[Service]
Type=simple
PIDFile=/run/tfserve_agent.pid
ExecStart=/bin/bash --login -c '/usr/local/bin/tensorflow_model_server --model_base_path=/root/resnet_v2_int8_NCHW/ --rest_api_port=8888'
User=root
Group=root
WorkingDirectory=/
Restart=always
[Install]
WantedBy=multi-user.target
EOH

gsutil cp gs://cloud-samples-data/dlvm/t4/model.tar.gz /root/model.tar.gz
tar -xzvf /root/model.tar.gz -C /root
# Reload systemd manager configuration
systemctl daemon-reload
# Enable tfserve service
systemctl --no-reload --now enable /lib/systemd/system/tfserve.service
