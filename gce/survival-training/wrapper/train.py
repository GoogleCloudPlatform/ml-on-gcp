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


def generate_trainer(hyperparameters):
  """Generates a callable which performs a single step of training when called.

  Args:
    1. hyperparameters - hyperparameters to train with.

  Returns:
    trainer callable, which performs a single step of training every time it is
    called and returns a JSON serializable representation of its state at the
    end
  """
  def _trainer():
    """Dummy callable.

    Args:
      None

    Returns:
      A random number between 0 and 1.
    """
    return random.random()

  return _trainer


def runner(
    trainer_initializer,
    job_dir,
    train_steps,
    checkpoint_steps,
    hyperparameters
    ):
  """Runs a training job.

  Args:
    trainer_initializer: Function which accepts hyperparameter dictionary as its
    only argument and returns a callable representing a single step of training.
    job_dir: Directory in which checkpoints should be stored.
    train_steps: Total number of steps for which training should be performed.
    checkpoint_steps: Training steps between checkpoints.
    hyperparameters: Dictionary containing hyperparameter specification for the
    training job.

  Returns:
    None

  Raises:
    ValueError: If hyperparameters are inconsistent with existing checkpoints in
    job_dir.
  """
  current_checkpoint_index = 0
  current_hyperparameters = copy.copy(hyperparameters)

  last_path, last_index = latest_checkpoint(get_checkpoints(job_dir))
  if last_index is not None:
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

  train_step = trainer_initializer(hyperparameters)

  def finished(step):
    """Returns True if job is complete and False otherwise."""
    if train_steps is None:
      return False
    else:
      return step > train_steps

  result = None
  # TODO(nkashy1): Add test for "up to N steps" rather than "additional N steps"
  current_step = current_checkpoint_index*checkpoint_steps + 1
  while not finished(current_step):
    result = train_step()

    if current_step%checkpoint_steps == 0:
      checkpoint_data = generate_checkpoint(
          current_checkpoint_index,
          hyperparameters,
          result
      )
      save_checkpoint(job_dir, current_checkpoint_index, checkpoint_data)
      current_checkpoint_index += 1

    current_step += 1

  checkpoint_data = generate_checkpoint(
      current_checkpoint_index,
      hyperparameters,
      result
  )
  save_checkpoint(job_dir, current_checkpoint_index, checkpoint_data)


def generate_checkpoint(step, hyperparameters, model_state):
  """Generates checkpoint contents.

  Args:
    step: Training step at which this checkpoint was generated.
    hyperparameters: Dictionary specifying the model hyperparameters.
    model_state: A JSON serializable representation of the model state.

  Returns:
    Dictionary representing the content to be checkpointed.
  """
  checkpoint_data = {
      "steps": step,
      "hyperparameters": hyperparameters,
      "model": model_state
  }
  return checkpoint_data


def get_checkpoints(job_dir):
  """Get all the checkpoints in a given directory.

  Args:
    job_dir: Directory containing checkpoints.

  Returns:
    List of paths to checkpoint files in the given directory.
  """
  checkpoint_glob = os.path.join(job_dir, "dummy-checkpoint-*.json")
  checkpoints = glob.glob(checkpoint_glob)
  return checkpoints


def latest_checkpoint(checkpoint_paths):
  """Returns the path to the most recently stored checkpoint from a list of
  checkpoints.

  Args:
    checkpoint_paths: List of paths to checkpoint files.

  Returns:
    Path to the most recent checkpoint from the provided list.
  """
  if not checkpoint_paths:
    return (None, None)

  checkpoint_indices = map(checkpoint_index, checkpoint_paths)
  indexed_checkpoints = zip(checkpoint_paths, checkpoint_indices)
  sorted_indexed_checkpoints = sorted(indexed_checkpoints, key=lambda p: p[1])
  return sorted_indexed_checkpoints[-1]


def checkpoint_index(checkpoint_path):
  """Returns the index of the checkpoint along a given path.

  Args:
    checkpoint_path: Path to a checkpoint file.

  Returns:
    Integer specifying which checkpoint the path represents. For example,
    dummy-checkpoint-173.json represents the 173rd checkpoint, and this function
    would return the integer 173.
  """
  checkpoint_file = os.path.basename(checkpoint_path)
  prefix = "dummy-checkpoint-"
  suffix = ".json"
  return int(checkpoint_file[len(prefix):-len(suffix)])


def load_checkpoint(checkpoint_path):
  """Loads the checkpoint object stored at a given path.

  Args:
    checkpoint_path: Path along which checkpoint is stored.

  Returns:
    Python dictionary representing the data serialized in the checkpoint.
  """
  with open(checkpoint_path, "r") as fp:
    checkpoint_data = json.load(fp)
  return checkpoint_data


def save_checkpoint(job_dir, index, checkpoint_data):
  """Serializes checkpoint data and stores it in a given directory.

  Args:
    job_dir: Directory in which to store checkpoint data.
    index: Ordinal index of the checkpoint.
    checkpoint_data: Data to be stored in the checkpoint. (Note: currently
    assumed to be JSON serializable.)

  Returns:
    The path to the saved checkpoint file.
  """
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
      "--train-steps",
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

  runner(
      generate_trainer,
      args.job_dir,
      args.train_steps,
      args.checkpoint_steps,
      hparams
  )
