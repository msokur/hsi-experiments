import utils

from cross_validators.cross_validator_old import CrossValidationOld
from cross_validators.cross_validator_spain import CrossValidatorSpain
from cross_validators.cross_validator_experiment import CrossValidatorExperiment
from cross_validators.cross_validator_postprocessing import CrossValidatorPostProcessing
import config

def get_cross_validator(*args, **kwargs):
    CROSS_VALIDATORS = {
        'cv_old': 'cv_old',
        'cv_spain': 'cv_spain',
        'cv_postprocessing': 'cv_postprocessing',
        'cv_experiment': 'cv_experiment'
    }

    if config.CROSS_VALIDATOR == 'cv_old':
        return CrossValidationOld(*args, **kwargs)
    if config.CROSS_VALIDATOR == 'cv_spain':
        return CrossValidatorSpain(*args, **kwargs)
    if config.CROSS_VALIDATOR == 'cv_postprocessing':
        return CrossValidatorPostProcessing(*args, **kwargs)
    if config.CROSS_VALIDATOR == 'cv_experiment':
        return CrossValidatorExperiment(*args, **kwargs)

    raise (f'Warning! {config.CROSS_VALIDATOR} specified wrong')



if __name__ == '__main__':
    
    try:
        cross_validator = get_cross_validator(config.bea_db, execution_flags = {
                "generate_whole_cubes": False,
                "get_predictions_for_whole_cubes": False,
                "count_predictions_for_labeled": False,
                "thr_ranges": [[0.001, 0.01, 20]]
            })
        execution_flags = cross_validator.get_execution_flags()
        execution_flags['cross_validation'] = False
        cross_validator.pipeline(execution_flags=execution_flags)

        #utils.send_tg_message('Mariia, operations in cross_validation.py are successfully completed!')
               
    except Exception as e:

        utils.send_tg_message(f'Mariia, ERROR!!!, In CV error {e}')
        
        raise e
