import os
import utils

from cross_validators.cross_validator_normal import CrossValidationNormal
from cross_validators.cross_validator_spain import CrossValidatorSpain
from cross_validators.cross_validator_experiment import CrossValidatorExperiment
from cross_validators.cross_validator_postprocessing import CrossValidatorPostProcessing
import config


def get_cross_validator(*args, **kwargs):
    if config.CROSS_VALIDATOR == 'cv_normal':
        return CrossValidationNormal(*args, **kwargs)
    if config.CROSS_VALIDATOR == 'cv_spain':
        return CrossValidatorSpain(*args, **kwargs)
    if config.CROSS_VALIDATOR == 'cv_postprocessing':
        return CrossValidatorPostProcessing(*args, **kwargs)
    if config.CROSS_VALIDATOR == 'cv_experiment':
        return CrossValidatorExperiment(*args, **kwargs)

    raise f'Warning! {config.CROSS_VALIDATOR} specified wrong'
    
def out_of_the_box():
    cross_validator = get_cross_validator('_CV_test')

    # cross validation pipeline consists of 2 parts:
    # (1) cross_validation
    # (2) evaluation
    # By default both these parts are executed (True)
    # With execution_flags it's possible not to execute 'cross_validation' or 'evaluation' with False
    execution_flags = cross_validator.get_execution_flags()
    execution_flags['cross_validation'] = False

    # After execution_flags we pass parameters for evaluation
    # To see all possible parameters, their meaning and possible combinations refer to comments in
    # evaluation/evaluation_base.py -> EvaluationBase -> save_predictions_and_metrics()
    # you can pass any parameters from save_predictions_and_metrics() except training_csv_path and npz_folder,
    # because they passed automatically
    cross_validator.pipeline(execution_flags=execution_flags,
                             thresholds_range=[[0.0001, 0.001, 20]],
                             save_predictions=True,
                             save_curves=False)

    #utils.send_tg_message('Mariia, operations in cross_validation.py are successfully completed!')

def test_postprocessing_all_models():
    config.CV_GET_CHECKPOINT_FROM_VALID = False

    for model, scaling_type, threshold, checkpoint in zip([  # 'CV_3d_inception',
        'CV_3d_inception_exclude1_all',
        'CV_3d_inception_svn_every_third',
        # 'CV_3d_svn_every_third',
        # 'CV_3d_sample_weights_every_third',
        # 'CV_3d_every_third',
        # 'CV_3d_inception_exclude1_every_third',
    ], [  # 'l2_norm',
        'svn_T', 'svn_T',
        # 'svn_T', 'svn_T',
        # 'l2_norm', 'svn_T'
    ],
            [  # 0.2111,
                0.0189, 0.0456,
                # 0.0367, 0.45,
                # 0.1556, 0.0456
            ],
            [  # 36,
                16, 18,
                # 16, 18,
                # 16, 16
            ]):

        config.RAW_NPZ_PATH = os.path.join('/work/users/mi186veva/data_3d', scaling_type)
        config.NORMALIZATION_TYPE = scaling_type

        if scaling_type == 'svn_T':
            thresholds_range = [threshold - (3 * (threshold / 2)), threshold, 20]
        else:
            thresholds_range = [threshold, threshold + (3 * (threshold / 2)), 20]

        cross_validator = get_cross_validator(
            model, execution_flags={
                "generate_whole_cubes": False,
                # by default if "whole" folder is empty than we generate whole cubes, otherwise we don't. But with generate_whole_cubes it's possible to forse generate
                "get_predictions_for_whole_cubes": False,
                # by default if there is no predictions_whole.npy in test/name/cp-0000 than we count predoctions for whole cubes, otherwise - we don't. But with get_predictions_for_whole_cubes it's possible to forse count
                "save_predictions_and_metrics_on_labeled": {
                    # for detailed documentation of params in this dictionary see documentation for evaluation/evaluation_base.py/EvaluationBase.save_predictions_and_metrics()
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
                    "median_filters_raw_list": [5, 11, 15],
                    "median_filters_range": None,
                    "thresholds_raw_list": None,
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

if __name__ == '__main__':
    
    try:
        #out_of_the_box()
        #test_postprocessing_all_models()

        cross_validator = get_cross_validator(
            config.database_abbreviation, cross_validation_type='algorithm_plain', execution_flags={
                "generate_whole_cubes": False,
                # by default if "whole" folder is empty than we generate whole cubes, otherwise we don't. But with generate_whole_cubes it's possible to forse generate
                "get_predictions_for_whole_cubes": False,
                # by default if there is no predictions_whole.npy in test/name/cp-0000 than we count predoctions for whole cubes, otherwise - we don't. But with get_predictions_for_whole_cubes it's possible to forse count
                "save_predictions_and_metrics_on_labeled": {
                    # for detailed documentation of params in this dictionary see documentation for evaluation/evaluation_base.py/EvaluationBase.save_predictions_and_metrics()
                    "save_predictions": False,
                    "metrics": {
                        'save_metrics': False,
                        'checkpoints_range': None,
                        'checkpoints_raw_list': None,
                        'thresholds_range': None,
                        'thresholds_raw_list': None,
                        'save_curves': False
                    }
                },
                "check": {  # what thresholds and median filter sizes to check
                    "median_filters_raw_list": [5, 11, 15, 21, 25],
                    "median_filters_range": None,
                    "thresholds_raw_list": None,
                    "thresholds_range": [0.0001, 0.001, 5]
                }
            })

        execution_flags = cross_validator.get_execution_flags()
        execution_flags['cross_validation'] = False
        cross_validator.pipeline(execution_flags=execution_flags)

        utils.send_tg_message(f'Mariia, Whole post-processing is successfully completed!')
               
    except Exception as e:

        utils.send_tg_message(f'Mariia, ERROR!!!, In CV error {e}')
        
        raise e


