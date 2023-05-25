import argparse

import config
from provider import get_trainer
import utils


def cross_validation_step(model_name, except_names=[]):
    trainer = get_trainer(model_name=model_name, except_indexes=except_names)
    trainer.train()


if __name__ == '__main__':

    try:

        parser = argparse.ArgumentParser(description='Process some integers.')

        parser.add_argument('--model_name', type=str)
        parser.add_argument('--except_names', type=str)

        args = parser.parse_args()

        print(f'Hi from CV! with {args.model_name} and {args.except_names}')

        cross_validation_step(model_name=args.model_name, except_names=args.except_names.split(';'))

    except Exception as e:

        utils.send_tg_message(f'{config.USER}, ERROR!!!, In CV error {e}')
        raise e
