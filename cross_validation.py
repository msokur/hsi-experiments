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


if __name__ == '__main__':
    
    try:
        cross_validator = get_cross_validator('_CV_5x5_svnT_med')

        # cross validation pipeline consists of 2 parts:
        # (1) cross_validation
        # (2) evaluation
        # By default both these parts are executed (True)
        # With execution_flags it's possible not to execute 'cross_validation' or 'evaluation' with False
        execution_flags = cross_validator.get_execution_flags()
        # execution_flags['cross_validation'] = False

        # After execution_flags we pass parameters for evaluation
        # To see all possible parameters, their meaning and possible combinations refer to comments in
        # evaluation/evaluation_base.py -> EvaluationBase -> save_predictions_and_metrics()
        # you can pass any parameters from save_predictions_and_metrics() except training_csv_path and npz_folder,
        # because they passed automatically
        cross_validator.pipeline(execution_flags=execution_flags,
                                 thresholds_range=[[0.0001, 0.001, 20]],
                                 save_predictions=True,
                                 save_curves=False)

        utils.send_tg_message(f'{config.USER}, operations in cross_validation.py are successfully completed!')
               
    except Exception as e:

        utils.send_tg_message(f'{config.USER}, ERROR!!!, In CV error {e}')
        
        raise e


#cross_validator = get_cross_validator('ExperimentALLHowManyValidPatExclude/ExperimentALLHowManyValidPatExclude_WF_C10_', execution_flags = {
'''cross_validator = get_cross_validator(
    config.database_abbreviation, execution_flags={
        "generate_whole_cubes": False,
        "get_predictions_for_whole_cubes": False,
        "count_predictions_for_labeled": False,
        "thr_ranges": [], #[[0.1, 0.6, 5]],
        'save_curves': True
    })
execution_flags = cross_validator.get_execution_flags()
execution_flags['cross_validation'] = False
cross_validator.pipeline(execution_flags=execution_flags)'''