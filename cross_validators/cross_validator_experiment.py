import os
import argparse
import json

from cross_validators.cross_validator_base import CrossValidatorBase


class CrossValidatorExperiment(CrossValidatorBase):
    def __init__(self, *args, **kwargs):

        # -------------------------parser
        parser = argparse.ArgumentParser(description='Process some integers.')

        parser.add_argument('--experiment_folder', type=str)
        parser.add_argument('--cv_name', type=str)
        parser.add_argument('--config_index', type=str)
        parser.add_argument('--test_path', type=str)
        parser.add_argument('--abbreviation', type=str)

        args = parser.parse_args()
        self.args = args
        super().__init__(*args, **kwargs)

        print(f'Hi from CV! with {args.experiment_folder}, {args.cv_name} and config_index {args.config_index}')
        root_folder = args.experiment_folder.split(config.SYSTEM_PATHS_DELIMITER)[-1]
        self.root_folder = root_folder

        # -------------------------metrics path

        metrics_saving_path = os.path.join(args.test_path, args.cv_name)
        if not os.path.exists(metrics_saving_path):
            os.mkdir(metrics_saving_path)
        self.metrics_saving_path = metrics_saving_path

        # -------------------------configs

        config.MODEL_NAME_PATHS.append(root_folder)

        configs = None
        with open(os.path.join(args.experiment_folder, 'combinations.json'), 'r') as json_file:
            data = json.load(json_file)
            configs = data[int(args.config_index)]
            print(configs)

        for key, value in configs.items():
            setattr(config, key, value)
        config.CV_RESTORE_VALID_PATIENTS_SEQUENCE = args.abbreviation.replace('WF', 'WT')

        config.BATCHED_PATH += '_' + args.cv_name

    def evaluation(self):
        # ------------------------testing

        search_path = os.path.join(self.project_folder, 'logs', self.root_folder, self.args.cv_name, '*.csv')
        csv_path = CrossValidatorBase.get_csv(search_path)

        self.save_predictions_and_metrics_for_checkpoint(0,
                                                         self.metrics_saving_path,
                                                         csv_path,
                                                         thr_ranges=[  # [0.1, 0.6, 10],
                                                             [0.4, 0.5, 100]],
                                                         execution_flags=[True])
