from tensorflow import keras

from models.model_randomness import get_initializers, set_tf_seed

import abc


class ModelBase:
    def __init__(self, input_shape: tuple, model_config: dict, num_of_output: int, name: str = None):
        self.input_shape = input_shape
        self.model_config = model_config
        self.num_of_output = num_of_output
        self.kernel_initializer, self.bias_initializer = get_initializers()
        self.name = name
        set_tf_seed()

    @abc.abstractmethod
    def get_model(self):
        pass

    @staticmethod
    def load_model(model_path: str, custom_objects=None) -> keras.Model:
        """
        Loads a keras model, return the model and set seed if necessary.

        :param model_path: The path to the model.
        :param custom_objects: A dict with custom metrics.

        Example
        -------
        custom_objects = {'F1_score': custom_metrics.F1_score}

        """
        set_tf_seed()
        model = keras.models.load_model(filepath=model_path, custom_objects=custom_objects)

        return model
