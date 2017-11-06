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

import json
import os
import random
import tempfile
import unittest

from . import train


class TrainTest(unittest.TestCase):

  def setUp(self):
    self.job_dir = tempfile.mkdtemp()
    self.num_checkpoints = 10
    self.checkpoint_files = []
    self.checkpoint_steps = 100

    # Note that hyperparameters are intended to be constant across checkpoints
    self.hyperparameter_1 = 17
    self.hyperparameter_2 = 3.14159

    for i in range(self.num_checkpoints):
      path = os.path.join(
          self.job_dir,
          "dummy-checkpoint-{}.json".format(i)
      )

      checkpoint_data = {
          "steps": i*self.checkpoint_steps,
          "hyperparameters": {
              "hyperparameter_1": self.hyperparameter_1,
              "hyperparameter_2": self.hyperparameter_2
          },
          "metric": random.random()
      }

      with open(path, "w") as fp:
        json.dump(checkpoint_data, fp)

      self.checkpoint_files.append(path)

    self.garbage_file = os.path.join(self.job_dir, "garbage")
    with open(self.garbage_file, "w") as gf:
      gf.write("garbage")

  def tearDown(self):
    os.remove(self.garbage_file)

    for path in self.checkpoint_files:
      os.remove(path)

    os.rmdir(self.job_dir)

  def test_get_checkpoints(self):
    checkpoints = train.get_checkpoints(self.job_dir)
    self.assertSetEqual(set(checkpoints), set(self.checkpoint_files))

  def test_checkpoint_index(self):
    indices = map(train.checkpoint_index, self.checkpoint_files)
    self.assertListEqual(indices, range(self.num_checkpoints))

  def test_latest_checkpoint_1(self):
    latest_checkpoint = train.latest_checkpoint(
        random.sample(self.checkpoint_files, self.num_checkpoints)
    )
    self.assertEqual(
        latest_checkpoint,
        (self.checkpoint_files[-1], self.num_checkpoints-1)
    )

  def test_latest_checkpoint_2(self):
    latest_checkpoint = train.latest_checkpoint([])
    self.assertIsNone(latest_checkpoint)

if __name__ == "__main__":
  unittest.main()
