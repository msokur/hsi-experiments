import numpy as np

import provider
import utils
import configuration.get_config as config
from configuration.get_config import telegram


    
def out_of_the_box():
    cross_validator = provider.get_cross_validator(typ=config.CONFIG_CV["TYPE"], cv_config=config.CONFIG_CV, paths=config.CONFIG_PATHS,
                                                   loader_config=config.CONFIG_DATALOADER)
    # cross validation pipeline consists of 2 parts:
    # (1) cross_validation
    # (2) evaluation
    # By default both these parts are executed (True)
    # With execution_flags it's possible not to execute 'cross_validation' or 'evaluation' with False
    execution_flags = config.CONFIG_CV["EXECUTION_FLAGS"]

    # After execution_flags we pass parameters for evaluation
    # To see all possible parameters, their meaning and possible combinations refer to comments in
    # evaluation/evaluation_base.py -> EvaluationBase -> save_predictions_and_metrics()
    # you can pass any parameters from save_predictions_and_metrics() except training_csv_path and npz_folder,
    # because they passed automatically
    cross_validator.pipeline(execution_flags=execution_flags,
                             # thresholds_range=[[0.0001, 0.001, 20]],    # specify thresholds if classification is binary
                             save_predictions=config.CONFIG_CV["SAVE_PREDICTION"],
                             save_curves=config.CONFIG_CV["SAVE_CURVES"])


def postprocessing_for_one_model():
    # Full documentation about post-processing: https://git.iccas.de/MaktabiM/hsi-experiments/-/wikis/Post-processing
    cross_validator = provider.get_cross_validator(# please refer to comments in cross_validators/cross_validator_postprocessing.py/CrossValidatorPostProcessing.blank_configuration() for detailed description of configuration params
                                          typ=config.CONFIG_CV["TYPE"],
                                          configuration={
                                                "generate_whole_cubes": False,
                                                "calculate_predictions_for_whole_cubes": False,
                                                "save_predictions_and_evaluate_on_labeled_samples": {
                                                    "save_predictions": False,
                                                    "metrics": {
                                                        'save_metrics': False,
                                                        'checkpoints': None,
                                                        'thresholds': np.round(np.linspace(0.0001, 0.001, 5), 4),
                                                        'save_curves': False
                                                    }
                                                },
                                                "postprocessing": {  
                                                    "MF_sizes": [31, 41, 51, 61], 
                                                    "thresholds": np.round(np.linspace(0.0001, 0.001, 5), 4)
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

        telegram.send_tg_message('Operations in cross_validation.py are successfully completed!')
               
    except Exception as e:

        telegram.send_tg_message(f'ERROR!!!, In CV error {e}')
        
        raise e
