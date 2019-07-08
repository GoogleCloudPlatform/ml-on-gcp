# Machine Learning on Google Cloud Platform - Example Zoo Tools

This directory contains tools that copies and modifies code examples from other public repositories so that they are readily runnable on AI Platform and tested.

## Usage

1. Register the samples in [`samples.yaml`](samples.yaml).

1. In python 2, run from the `tools` directory:

	```
	python process.py
	```

## Sample configuration

The tool parses the [`samples.yaml`](samples.yaml) file to create samples.  For example,

```yaml
samples:
  - org: tensorflow
    repository: probability
    branch: "r0.6"
    source_path: tensorflow_probability/examples
    source_name: bayesian_neural_network.py
    requires:
      - "seaborn==0.9.0"
    tfgfile_wrap:
      - plot_weight_posteriors
      - plot_heldout_prediction
    args:
      - "--fake_data"
      - "--max_steps=5"
      - "--viz_steps=5"
    artifact: weights.png
    wait_time: 600
```

creates:

```
.
└── tensorflow
    └── probability
        └── bayesian_neural_network
            ├── README.md
            ├── bayesian_neural_network_test.py
            ├── config.yaml
            ├── setup.py
            ├── submit.sh
            └── trainer
                ├── __init__.py
                ├── bayesian_neural_network.py
                └── tfgfile_wrapper.py
```

In this case only the example script `bayesian_neural_network.py` file is copied over from the source, specified by the `org`, `repository`, and `branch` fields.  The example script's location is specified by the `source_path` and `source_name` fields.  All other files are generated from the templates in `tools/templates`.

The `requires` field specified additional packages to be added to the generated `setup.py` file.

The `tfgfile_wrap` function wraps functions in the example script that write to local disk, and write to `job-dir` specified in in `submit.sh` on Google Cloud Storage instead.  This allows the tests to inspect artifacts when the job is running on AI Platform Training.

The `wait_time` field specifies how long the test will wait before checking for artifacts, and the `artifact` field specifies a portion of the artifact filename that must be observed for the job to be considered successful.  The `args` list will be included in the generated `submit.sh`, and should be used to specify a small test dataset (testingi s done by running `submit.sh`).
