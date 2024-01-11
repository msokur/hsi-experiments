import pytest
import os

from tensorflow.keras.layers import Dense, Conv1D, Conv3D, Dropout, Flatten, InputLayer, MaxPooling1D, MaxPooling3D, \
    Concatenate, BatchNormalization, Activation
from keras.layers.core.tf_op_layer import TFOpLambda
from keras.layers.core.reshape import Reshape

from models.paper_model import PaperModel1D, PaperModel3D
from models.inception_model import InceptionModel1D, InceptionModel3D


@pytest.fixture(scope="session", autouse=True)
def no_random():
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    yield


PAPER_MODEL_CONFIGURATION = {"DROPOUT": 0.01}

INCEPTION_MODEL_CONFIGURATION_1 = {
    "WITH_BATCH_NORM": False,
    "INCEPTION_FACTOR": 8,
    "DROPOUT": 0.1
}

INCEPTION_MODEL_CONFIGURATION_2 = {
    "WITH_BATCH_NORM": True,
    "INCEPTION_FACTOR": 8,
    "DROPOUT": 0.1
}

GET_MODEL_DATA = [(PaperModel1D, (92,), 2, PAPER_MODEL_CONFIGURATION,
                   [InputLayer, TFOpLambda, Conv1D, Conv1D, Conv1D, Conv1D, Conv1D, Conv1D, Conv1D, Conv1D, Flatten,
                    Dropout, Dense]),
                  (PaperModel1D, (62,), 2, PAPER_MODEL_CONFIGURATION,
                   [InputLayer, TFOpLambda, Conv1D, Conv1D, Conv1D, Conv1D, Conv1D, Conv1D, Flatten, Dropout, Dense]),
                  (PaperModel3D, (3, 3, 92), 3, PAPER_MODEL_CONFIGURATION,
                   [InputLayer, TFOpLambda, Conv3D, Conv3D, Reshape, Conv1D, Conv1D, Flatten, Dropout, Dense]),
                  (PaperModel3D, (5, 5, 92), 3, PAPER_MODEL_CONFIGURATION,
                   [InputLayer, TFOpLambda, Conv3D, Conv3D, Conv3D, Conv3D, Reshape, Conv1D, Conv1D, Conv1D, Conv1D,
                    Flatten, Dropout, Dense]),
                  (PaperModel3D, (11, 11, 92), 20, PAPER_MODEL_CONFIGURATION,
                   [InputLayer, TFOpLambda, Conv3D, Conv3D, Conv3D, Conv3D, Conv3D, Conv3D, Conv3D, Conv3D, Reshape,
                    Conv1D, Conv1D, Conv1D, Conv1D, Flatten, Dropout, Dense]),
                  (InceptionModel1D, (92,), 2, INCEPTION_MODEL_CONFIGURATION_1,
                   [InputLayer, TFOpLambda, Conv1D, Conv1D, MaxPooling1D, Conv1D, Conv1D, Conv1D, Conv1D, Concatenate,
                    Flatten, Dropout, Dense]),
                  (InceptionModel3D, (3, 3, 92), 4, INCEPTION_MODEL_CONFIGURATION_1,
                   [InputLayer, TFOpLambda, Conv3D, Conv3D, MaxPooling3D, Conv3D, Conv3D, Conv3D, Conv3D,
                    Concatenate, Flatten, Dropout, Dense]),
                  (InceptionModel1D, (92,), 2, INCEPTION_MODEL_CONFIGURATION_2,
                   [InputLayer, TFOpLambda, Conv1D, Conv1D, BatchNormalization, BatchNormalization, Activation,
                    Activation, MaxPooling1D, Conv1D, Conv1D, Conv1D, Conv1D, BatchNormalization, BatchNormalization,
                    BatchNormalization, BatchNormalization, Activation, Activation, Activation, Activation,
                    Concatenate, Flatten, Dropout, Dense]),
                  (InceptionModel3D, (3, 3, 92), 4, INCEPTION_MODEL_CONFIGURATION_2,
                   [InputLayer, TFOpLambda, Conv3D, Conv3D, BatchNormalization, BatchNormalization, Activation,
                    Activation, MaxPooling3D, Conv3D, Conv3D, Conv3D,  Conv3D, BatchNormalization, BatchNormalization,
                    BatchNormalization, BatchNormalization, Activation, Activation, Activation, Activation,
                    Concatenate, Flatten, Dropout, Dense])]


@pytest.mark.parametrize("model_class,spec,labels,config,result_layers", GET_MODEL_DATA)
def test_get_model_input(model_class, spec, labels, config, result_layers):
    model = model_class(input_shape=spec, config=config, num_of_output=labels).get_model()

    assert len(model.inputs) == 1
    assert model.input_shape == ((None,) + spec)


@pytest.mark.parametrize("model_class,spec,labels,config,result_layers", GET_MODEL_DATA)
def test_get_model_layer(model_class, spec, labels, config, result_layers):
    model = model_class(input_shape=spec, config=config, num_of_output=labels).get_model()

    assert len(model.layers) == len(result_layers)
    for layer, result_layer in zip(model.layers, result_layers):
        assert isinstance(layer, result_layer)


@pytest.mark.parametrize("model_class,spec,labels,config,result_layers", GET_MODEL_DATA)
def test_get_model_output(model_class, spec, labels, config, result_layers):
    model = model_class(input_shape=spec, config=config, num_of_output=labels).get_model()

    assert len(model.outputs) == 1
    assert model.output_shape == (None, 1 if labels == 2 else labels)
