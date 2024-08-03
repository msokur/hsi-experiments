import numpy as np
import provider
from configuration.get_config import telegram
from configuration.keys import CrossValidationKeys as CVK, DataLoaderKeys as DLK
from evaluation.optimal_parameters import OptimalThreshold


def out_of_the_box(config):
    cross_validator = provider.get_cross_validator(config=config, typ=config.CONFIG_CV[CVK.TYPE])
    # cross validation pipeline consists of 2 parts:
    # (1) cross_validation
    # (2) evaluation
    # By default both these parts are executed (True)
    # With execution_flags it is possible not to execute 'cross_validation' or 'evaluation' with False
    execution_flags = config.CONFIG_CV[CVK.EXECUTION_FLAGS]

    # After execution_flags we pass parameters for evaluation
    # To see all possible parameters, their meaning and possible combinations refer to comments in
    # evaluation/evaluation_base.py -> EvaluationBase -> save_predictions_and_metrics()
    # you can pass any parameters from save_predictions_and_metrics() except training_csv_path and npz_folder,
    # because they passed automatically
    thresholds = None
    if len(config.CONFIG_DATALOADER[DLK.LABELS_TO_TRAIN]) <= 2:
        thresholds = np.round(np.linspace(0.001, 0.6, 100), 4)  # specify thresholds if classification is binary
    cross_validator.pipeline(execution_flags=execution_flags,
                             thresholds=thresholds)

    optimal_threshold_finder = OptimalThreshold(config, prints=False)
    optimal_threshold_finder.add_additional_thresholds_if_needed(cross_validator)


def postprocessing_for_one_model(config):
    # Full documentation about post-processing: https://git.iccas.de/MaktabiM/hsi-experiments/-/wikis/Post-processing
    cross_validator = provider.get_cross_validator(
        # please refer to comments in
        # cross_validators/cross_validator_postprocessing.py/CrossValidatorPostProcessing.blank_configuration()
        # for detailed description of configuration params
        config=config,
        typ=config.CONFIG_CV[CVK.TYPE],
        configuration={
            "generate_whole_cubes": False,
            "calculate_predictions_for_whole_cubes": False,
            "save_predictions_and_evaluate_on_labeled_samples": {
                "save_predictions": False,
                "metrics": {
                    'save_metrics': False,
                    'checkpoints': None,
                    'thresholds': None,  # np.round(np.linspace(0.0001, 0.001, 5), 4),
                    'save_curves': False
                }
            },
            "postprocessing": {
                "MF_sizes": [31, 41, 51, 61],
                "thresholds": np.round(np.linspace(0.0001, 0.001, 5), 4)
            }
        })

    # cross_validators pipeline consist of 2 steps:
    # 'cross_validation' (where models are trained) and 'evaluation'.
    # In this case we only want to apply post-processing (evaluation) on existing model
    # (that's why 'cross_validation' is set to False)
    execution_flags = cross_validator.get_execution_flags()
    execution_flags['cross_validation'] = False
    cross_validator.pipeline(execution_flags=execution_flags)


if __name__ == '__main__':
    try:
        import configuration.get_config as configuration

        out_of_the_box(configuration)
        # postprocessing_for_one_model(config)

        telegram.send_tg_message(f'Operations in cross_validation.py for {configuration.CONFIG_CV[CVK.NAME]} '
                                 f'are successfully completed!')

    except Exception as e:

        telegram.send_tg_message(f'ERROR!!!, In CV {configuration.CONFIG_CV[CVK.NAME]} error {e}')

        raise e
