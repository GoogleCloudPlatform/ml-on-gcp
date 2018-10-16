import time

from cmle_utils import get_models, get_model_versions, train_model, deploy_model, set_default, predict_json, build_and_upload_package, create_package


class CMLETraining(object):
    def __init__(self, job_dir=None, model_fn=None, train_input_fn=None, parse_args=None, train=None):
        self.package_path = None
        self.job_dir = job_dir

        self.model_fn = model_fn
        self.train_input_fn = train_input_fn
        self.parse_args = parse_args
        self.train = train

        # TODO: add sanity check of the return types of the methods


    # TODO: make the object aware of whether this has been run.
    def _create_package(self):
        create_package(self.model_fn, self.train_input_fn, self.parse_args, self.train)


    def _build_and_upload_package(self, gcs_directory):
        self.package_path = build_and_upload_package(gcs_directory)


    def _format_args(self, args_dict):
        args = []
        for k, v in args_dict.iteritems():
            args.append('--{k} {v}'.format(k=k, v=v))

        return args


    def _training_input(self, args):
        args = args or []
        training_input = {
            'scaleTier': 'BASIC',
            'packageUris': [self.package_path],
            'pythonModule': 'trainer.task',
            'region': 'us-central1',
            'jobDir': self.job_dir,
            'runtimeVersion': '1.10',
            'args': args
        }

        return training_input


    def local_train(self):
        args = self.parse_args()
        print(args)

        return self.train(args)


    # TODO: get project_id from default credentials
    # TODO: investigate auto-deployment (tricky since we'll need a thread pinging the training job and deploy the model when it is finished)
    def cloud_train(self, project_id, job_id=None, training_input=None, args_dict=None):
        assert self.job_dir is not None

        if self.package_path is None:
            self._create_package()
            self._build_and_upload_package(self.job_dir)

        if args_dict is not None:
            args = self._format_args(args_dict)

        base_training_input = self._training_input(args)

        if training_input is not None:
            base_training_input.update(training_input)

        training_input = base_training_input

        # job id must start with a letter
        job_id = job_id or 'test_{}'.format(str(int(time.time())))

        job_spec = {
            'jobId': job_id,
            'trainingInput': training_input
        }

        print('Job spec: {}'.format(job_spec))

        return train_model(project_id, job_spec)



