import numpy as np
from tqdm import tqdm

from cross_validators.cross_validator_base import CrossValidatorBase

class CrossValidationOld(CrossValidatorBase):
    def __init__(self):
        super().__init__("")

    def evaluation(self):
        return

    def compare_checkpoints(self, rng, save_path_, results_file):
        rg = np.linspace(rng[0], rng[1], rng[2]).astype(int)
        checkpoints = [f'cp-{i:04d}' for i in rg]
        print('Checkpoints: ', checkpoints)

        for checkpoint in tqdm(checkpoints):
            self.save_predictions_and_metrics_for_checkpoint(checkpoint, save_path_, results_file)
