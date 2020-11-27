
import argparse
import os
import tensorflow as tf
from tf_trainer import Trainer
from hparams import hparams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hparams', type=str,
                        help='Comma separated list of "name=value" pairs.')
    args = parser.parse_args()

    from hparams import hparams

    if args.hparams is not None:
        hparams.parse(args.hparams)

    from model import Model

    trainer = Trainer(hparams,
                      training_dir_path=os.path.join(hparams.get('training_dir_path'),
                                                     hparams.get('model_name', '')))

    trainer.set_model(Model, var_scope='') \
        .freeze(input_getter=lambda: [tf.placeholder(tf.float32, [None] + hparams.get('datasets_images_size') + [3], name='input_images')],
                frozen_name=hparams.get('model_name', 'graph.frozen'), verbose=True)


if __name__ == '__main__':
    main()
