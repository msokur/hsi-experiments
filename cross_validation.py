from configuration import get_config as conf
import provider_dyn

import numpy as np

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


def postprocessing_for_one_model():
    # Full documentation about post-processing: https://git.iccas.de/MaktabiM/hsi-experiments/-/wikis/Post-processing
    cross_validator = get_cross_validator(config.database_abbreviation,   
                                          # please refer to comments in cross_validators/cross_validator_postprocessing.py/CrossValidatorPostProcessing.blank_configuration() for detailed description of configuration params
                                          configuration={
                                                "generate_whole_cubes": False,
                                                "calculate_predictions_for_whole_cubes": False,
                                                "save_predictions_and_evaluate_on_labeled_samples": {
                                                    "save_predictions": False,
                                                    "metrics": {
                                                        'save_metrics': False,
                                                        'checkpoints': None,
                                                        'thresholds': None, #np.round(np.linspace(0.0001, 0.001, 5), 4),
                                                        'save_curves': False
                                                    }
                                                },
                                                "postprocessing": {  
                                                    "MF_sizes": [31, 41, 51, 61], 
                                                    "thresholds": None, #np.round(np.linspace(0.0001, 0.001, 5), 4)
                                                }
                                          })

    execution_flags = cross_validator.get_execution_flags()
    execution_flags['cross_validation'] = False   # cross_validators pipeline consist of 2 steps: 'cross_validation' (where models are trained) and 'evaluation'. In this case we only want to apply post-processing (evaluation) on existing model (that's why 'cross_validation' is set to False)
    cross_validator.pipeline(execution_flags=execution_flags)
if __name__ == '__main__':
    from datetime import date
    try:
	#out_of_the_box()        
        postprocessing_for_one_model()

        '''cross_validator = provider_dyn.get_cross_validator(typ=conf.CV["TYPE"], cv_config=conf.CV, paths=conf.PATHS,
                                                           loader_config=conf.DATALOADER)
        # cross validation pipeline consists of 2 parts:
        # (1) cross_validation
        # (2) evaluation
        # By default both these parts are executed (True)
        # With execution_flags it's possible not to execute 'cross_validation' or 'evaluation' with False
        execution_flags = conf.CV["EXECUTION_FLAGS"]
        # execution_flags['cross_validation'] = False

        # After execution_flags we pass parameters for evaluation
        # To see all possible parameters, their meaning and possible combinations refer to comments in
        # evaluation/evaluation_base.py -> EvaluationBase -> save_predictions_and_metrics()
        # you can pass any parameters from save_predictions_and_metrics() except training_csv_path and npz_folder,
        # because they passed automatically
        cross_validator.pipeline(execution_flags=execution_flags,
                                 # thresholds_range=[[0.0001, 0.001, 20]],
                                 save_predictions=conf.CV["SAVE_PREDICTION"],
                                 save_curves=conf.CV["SAVE_CURVES"])'''

        # conf.telegram.send_tg_message('operations in cross_validation.py are successfully completed!')
               
    except Exception as e:

        utils.send_tg_message(f'Mariia, ERROR!!!, In CV error {e}')
        
        raise e
