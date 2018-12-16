# Copyright 2018 Google Inc. All Rights Reserved.
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
"""Converts TF SavedModel to the TensorRT enabled graph"""

import argparse
import tensorflow.contrib.tensorrt as trt


parser = argparse.ArgumentParser(
	description="Converts SavedModel to the TensorRT enabled graph.")

parser.add_argument("--input_model_dir",
										required=True)
parser.add_argument("--output_model_dir",
										required=True)
parser.add_argument("--batch_size",
										type=int,
										required=True)
parser.add_argument("--precision_mode",
										choices=["FP32", "FP16", "INT8", "INT4"],
										required=True)
args = parser.parse_args()

trt.create_inference_graph(
	None, None, max_batch_size=args.batch_size,
	input_saved_model_dir=args.input_model_dir,
	output_saved_model_dir=args.output_model_dir,
	precision_mode=args.precision_mode)
