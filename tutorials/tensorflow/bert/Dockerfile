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
# Build Docker container:
# docker build --build-arg LD_LIBRARY_PATH=$LD_LIBRARY_PATH -t gcr.io/<project>/tensorrt_bert_sample:staging ./
#
FROM gcr.io/deeplearning-platform-release/base-cpu
FROM gcr.io/dpe-cloud-mle/tensorrt_bert_sample:dev

# AI Hub Plugin
COPY --from=0 /root/miniconda3/lib/python3.7/site-packages/jupyter_aihub_deploy_extension /usr/local/lib/python3.6/dist-packages/jupyter_aihub_deploy_extension
RUN pip install requests

COPY ./script.sh /usr/local/bin/start.sh
ARG LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/tensorrt/lib

EXPOSE 8080
ENTRYPOINT /usr/local/bin/start.sh


