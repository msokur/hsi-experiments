import numpy as np
from tqdm import tqdm
import os

import config
from cross_validators.cross_validator_base import CrossValidatorBase
from evaluation.evaluation_binary import EvaluationBinary
from evaluation.evaluation_multiclass import EvaluationMulticlass

class CrossValidationNormal(CrossValidatorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluation(self, **kwargs):
        training_csv_path = self.get_csv(os.path.join(self.project_folder, 'logs', self.name))
        print(training_csv_path)

        if len(config.LABELS_OF_CLASSES_TO_TRAIN) > 2:
            evaluator = EvaluationMulticlass(config.database_abbreviation)
        else:
            evaluator = EvaluationBinary(config.database_abbreviation)

        evaluator.save_predictions_and_metrics(training_csv_path=training_csv_path,
                                               npz_folder=config.RAW_NPZ_PATH,
                                               **kwargs)

    def compare_checkpoints(self, rng, save_path_, results_file):
        rg = np.linspace(rng[0], rng[1], rng[2]).astype(int)
        checkpoints = [f'cp-{i:04d}' for i in rg]
        print('Checkpoints: ', checkpoints)

        for checkpoint in tqdm(checkpoints):
            self.save_predictions_and_metrics_for_checkpoint(checkpoint, save_path_, results_file)
