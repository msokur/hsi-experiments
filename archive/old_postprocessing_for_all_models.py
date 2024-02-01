from archive import config
import os
from provider import get_cross_validator

# old function for cross_validation.py to test models for post-processing papers
def postprocessing_test_all_models():
    config.CV_GET_CHECKPOINT_FROM_VALID = False

    for model, scaling_type, threshold, checkpoint in zip([
        'CV_3d_inception',
        # 'CV_3d_inception_exclude1_all',
        # 'CV_3d_inception_svn_every_third',
        # 'CV_3d_svn_every_third',
        # 'CV_3d_sample_weights_every_third',
        # 'CV_3d_every_third',
        # 'CV_3d_inception_exclude1_every_third',
    ], ['l2_norm',
        # 'svn_T',
        # 'svn_T',
        # 'svn_T',
        # 'svn_T',
        # 'l2_norm',
        # 'svn_T'
        ],
            [0.2111,
             # 0.0189,
             # 0.0456,
             # 0.0367,
             # 0.45,
             # 0.1556,
             # 0.0456
             ],
            [36,
             # 16,
             # 18,
             # 16,
             # 18,
             # 16,
             # 16
             ]):

        config.CONFIG_PATHS["RAW_NPZ_PATH"] = os.path.join('/work/users/mi186veva/data_3d', scaling_type)
        config.NORMALIZATION_TYPE = scaling_type

        if scaling_type == 'svn_T':
            thresholds_range = [0.00001, threshold, 20]
        else:
            thresholds_range = [threshold - (threshold / 2), threshold + (threshold / 2), 20]
            # thresholds_range = [threshold, 2 * threshold, 20]

        cross_validator = get_cross_validator(
            model, cross_validation_type='algorithm_with_threshold', configuration={
                "generate_whole_cubes": False,
                # by default if "whole" folder is empty then we generate whole cubes,
                # otherwise we don't. But with generate_whole_cubes it is possible to force generate
                "calculate_predictions_for_whole_cubes": False,
                # by default if there is no predictions_whole.npy in metrics/name/cp-0000 than we count predictions
                # for whole cubes, otherwise - we don't.
                # But with calculate_predictions_for_whole_cubes it's possible to force count
                "save_predictions_and_evaluate_on_labeled_samples": {
                    # for detailed documentation of params in this dictionary see documentation for
                    # evaluation/evaluation_base.py/EvaluationBase.save_predictions_and_metrics()
                    "save_predictions": False,
                    "metrics": {
                        'save_metrics': False,
                        'checkpoints_range': None,
                        'checkpoints_raw_list': [checkpoint],
                        'thresholds_range': None,
                        'thresholds_raw_list': None,
                        'save_curves': False
                    }
                },
                "check": {  # what thresholds and median filter sizes to check
                    "median_filters_raw_list": [5, 25, 51],
                    # [31, 35, 41, 45, 51, 55, 61, 65], #[5, 11, 15, 21, 25, 31, 35, 41, 45, 51, 55, 61],
                    "median_filters_range": None,
                    "thresholds_raw_list": None,  # [0.1056, 0.2111, 0.3166],
                    "thresholds_range": thresholds_range

                }
            })
        cross_validator.evaluator.checkpoint_basename += scaling_type + '_'

        cross_validator.saving_folder_with_checkpoint = os.path.join(cross_validator.saving_folder,
                                                                     f'cp-{scaling_type}_{checkpoint:04d}')

        execution_flags = cross_validator.get_execution_flags()
        execution_flags['cross_validation'] = False
        cross_validator.pipeline(execution_flags=execution_flags)

        # utils.send_tg_message(f'Mariia, Post-processing for {model} is successfully completed!')