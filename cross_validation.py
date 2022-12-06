from configuration import get_config as conf
import provider_dyn


if __name__ == '__main__':
    from datetime import date
    try:

        cross_validator = provider_dyn.get_cross_validator(typ=conf.CV["TYPE"], cv_config=conf.CV, paths=conf.PATHS,
                                                           loader_config=conf.DATALOADER)
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
                                 # thresholds_range=[[0.0001, 0.001, 20]],
                                 save_predictions=True,
                                 save_curves=False)

        conf.telegram.send_tg_message('operations in cross_validation.py are successfully completed!')
               
    except Exception as e:

        conf.telegram.send_tg_message(f'ERROR!!!, In CV error {e}')
        
        raise e
