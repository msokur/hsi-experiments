from tensorflow.keras.optimizers import Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSprop, SGD

OPTIMIZER = {"adadelta": Adadelta,
             "adagrad": Adagrad,
             "adam": Adam,
             "adamax": Adamax,
             "ftrl": Ftrl,
             "nadam": Nadam,
             "rmsprop": RMSprop,
             "sgd": SGD
             }
