import os

import provider
from cross_validators.cross_validator_base import CrossValidatorBase


class CrossValidationNormal(CrossValidatorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluation(self, **kwargs):
        training_csv_path = self.get_csv(os.path.join(self.project_folder, "logs", self.CONFIG_CV["NAME"]))
        print('training_csv_path', training_csv_path)

        evaluator = provider.get_evaluation(config=self.config, labels=self.CONFIG_DATALOADER["LABELS_TO_TRAIN"],
                                            name=self.CONFIG_CV["DATABASE_ABBREVIATION"])

        evaluator.save_predictions_and_metrics(training_csv_path=training_csv_path,
                                               npz_folder=self.CONFIG_PATHS["RAW_NPZ_PATH"],
                                               **kwargs)
