import numpy as np
from tqdm import tqdm
import os

import provider_dyn
from cross_validators.cross_validator_base import CrossValidatorBase


class CrossValidationNormal(CrossValidatorBase):
    def __init__(self):
        super().__init__()

    def evaluation(self, **kwargs):
        training_csv_path = self.get_csv(os.path.join(self.project_folder, "logs", self.cv["NAME"]))
        print(training_csv_path)

        evaluator = provider_dyn.get_evaluation(labels=self.loader["LABELS_TO_TRAIN"],
                                                name=self.cv["DATABASE_ABBREVIATION"])

        evaluator.save_predictions_and_metrics(training_csv_path=training_csv_path,
                                               npz_folder=self.paths["RAW_NPZ_PATH"],
                                               **kwargs)

    def compare_checkpoints(self, rng, save_path_, results_file):
        rg = np.linspace(rng[0], rng[1], rng[2]).astype(int)
        checkpoints = [f'cp-{i:04d}' for i in rg]
        print('Checkpoints: ', checkpoints)

        for checkpoint in tqdm(checkpoints):
            self.save_predictions_and_metrics_for_checkpoint(checkpoint, save_path_, results_file)
