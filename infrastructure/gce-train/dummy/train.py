# Copyright 2017 Google Inc. All Rights Reserved.
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

import argparse
import copy
import glob
import json
import os
import random
import time


def dummy_trainer(job_dir, total_steps, checkpoint_steps, hyperparameters):
  current_checkpoint_index = 0
  current_hyperparameters = copy.copy(hyperparameters)

  last_checkpoint = latest_checkpoint(get_checkpoints(job_dir))
  if last_checkpoint is not None:
    last_path, last_index = last_checkpoint

    current_checkpoint_index = last_index + 1

    last_data = load_checkpoint(last_path)
    last_hp = last_data.get("hyperparameters")
    for hyperparameter in current_hyperparameters:
      if (current_hyperparameters[hyperparameter] is not None and
          current_hyperparameters[hyperparameter] != last_hp[hyperparameter]):
        raise ValueError(
            "Inconsistent values for {}: ".format(hyperparameter) +
            "command line -- {}, checkpoint -- {}".format(
                hyperparameters[hyperparameter],
                last_data[hyperparameter]
            )
        )

    current_hyperparameters = last_hp

  def finished(step):
    if total_steps is None:
      return False
    else:
      return step > total_steps

  i = 1
  while not finished(i):
    if i%checkpoint_steps == 0:
      checkpoint_data = generate_checkpoint(
          current_checkpoint_index,
          hyperparameters
      )
      save_checkpoint(job_dir, current_checkpoint_index, checkpoint_data)
      current_checkpoint_index += 1

    i += 1

  checkpoint_data = generate_checkpoint(
      current_checkpoint_index,
      hyperparameters
  )
  save_checkpoint(job_dir, current_checkpoint_index, checkpoint_data)


def generate_checkpoint(step, hyperparameters):
  checkpoint_data = {
      "steps": step,
      "hyperparameters": hyperparameters,
      "state": random.random()
  }
  return checkpoint_data


def get_checkpoints(job_dir):
  checkpoint_glob = os.path.join(job_dir, "dummy-checkpoint-*.json")
  checkpoints = glob.glob(checkpoint_glob)
  return checkpoints


def latest_checkpoint(checkpoint_paths):
  if not checkpoint_paths:
    return None

  checkpoint_indices = map(checkpoint_index, checkpoint_paths)
  indexed_checkpoints = zip(checkpoint_paths, checkpoint_indices)
  sorted_indexed_checkpoints = sorted(indexed_checkpoints, key=lambda p: p[1])
  return sorted_indexed_checkpoints[-1]


def checkpoint_index(checkpoint_path):
  checkpoint_file = os.path.basename(checkpoint_path)
  prefix = "dummy-checkpoint-"
  suffix = ".json"
  return int(checkpoint_file[len(prefix):-len(suffix)])


def load_checkpoint(checkpoint_path):
  with open(checkpoint_path, "r") as fp:
    checkpoint_data = json.load(fp)
  return checkpoint_data


def save_checkpoint(job_dir, index, checkpoint_data):
  checkpoint_file = "dummy-checkpoint-{}.json".format(index)
  checkpoint_path = os.path.join(job_dir, checkpoint_file)
  with open(checkpoint_path, "w") as fp:
    json.dump(checkpoint_data, fp)
  return checkpoint_path


if __name__ == "__main__":
  parser = argparse.ArgumentParser("Dummy trainer")

  parser.add_argument(
      "--job-dir",
      help="Directory where checkpoints and checkpoint metadata will be written"
  )
  parser.add_argument(
      "--checkpoint-steps",
      type=int,
      help="Number of steps per checkpointing operation"
  )
  parser.add_argument(
      "--total-steps",
      type=int,
      default=None,
      help=("Total number of steps that you would like to train for -- "
            "trains forever if this argument is not specified")
  )
  parser.add_argument(
      "--hyperparameter-1",
      type=int,
      required=False,
      help="Generic integer hyperparameter for dummy model"
  )
  parser.add_argument(
      "--hyperparameter-2",
      type=float,
      required=False,
      help="Generic floating point hyperparameter for dummy model"
  )

  args = parser.parse_args()

  hparams = {
      "hyperparameter_1": args.hyperparameter_1,
      "hyperparameter_2": args.hyperparameter_2
  }

  dummy_trainer(args.job_dir, args.total_steps, args.checkpoint_steps, hparams)
