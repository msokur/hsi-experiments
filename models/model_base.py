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
