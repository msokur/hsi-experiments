import abc

from keras import layers, Model
import numpy as np

from models.kt_hypermodel_base import KtModelBase
from models.model_randomness import get_dropout

from configuration.keys import TunerModelKeys as TMK


class PaperTunerModelBase(KtModelBase):
    def _model(self, input_layer: layers.Input, net: layers.Reshape):
        net = self._get_block(net=net)

        net = layers.Flatten(name="flatt")(net)

        for fc in range(self.hp.Int("num_fc", **self.model_config[TMK.NUM_DENSE])):
            net = layers.Dense(self.hp.Int(f"fc_{fc}", **self.model_config[TMK.DENSE_UNITS]),
                               activation=self._get_activations(name=f"fc_{fc}"), name=f"fc_{fc}")(net)

            if self.hp.Boolean(f"fc_{fc}_dropout"):
                net = get_dropout(net=net,
                                  dropout_value=self.hp.Float(f"fc_{fc}_dropout_value",
                                                              **self.model_config[TMK.DROPOUT]),
                                  name=f"fc_{fc}_dropout")

        activation = "sigmoid"
        number = 1
        if self.num_of_classes > 2:
            activation = None
            number = self.num_of_classes

        result = layers.Dense(number, activation=activation, kernel_initializer=self.kernel_initializer,
                              bias_initializer=self.bias_initializer)(net)

        model = Model(
            inputs=[input_layer],
            outputs=[result]
        )

        return model

    @abc.abstractmethod
    def _get_block(self, net):
        pass

    def _get_conv1d(self, net, name, kernel_size=3, strides=2):
        net = layers.Conv1D(filters=self.hp.Int(f"{name}_1", **self.model_config[TMK.CONV_1D_FILTERS]),
                            kernel_size=kernel_size, padding='valid', activation='relu',
                            kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer,
                            name=f"{name}_1")(net)
        net = layers.Conv1D(filters=self.hp.Int(f"{name}_2", **self.model_config[TMK.CONV_1D_FILTERS]),
                            kernel_size=kernel_size if net.shape[-2] > 1 else 1, strides=strides,
                            padding='valid', activation='relu',
                            kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer,
                            name=f"{name}_2")(net)

        return net


class PaperTunerModel3D(PaperTunerModelBase):
    def _get_block(self, net):
        max_conv_layer = int(self.input_shape[0] / 2)

        for r in range(self.hp.Int("num_conv3d", min_value=1, max_value=max_conv_layer)):
            if net.shape[-3] == 1:
                break
            elif net.shape[-3] == 2:
                net = self.__get_conv3d(net=net, name=f"conv_3d_{r}", kernel_size=2, strides=1)
                break
            else:
                net = self.__get_conv3d(net=net, name=f"conv_3d_{r}")

            net = self._wrap_layer(layer=net, name=f"conv_3d_{r}")

        net = layers.Reshape((np.prod(net.shape[-4:-1]), net.shape[-1]))(net)

        for r in range(self.hp.Int("num_conv1d", min_value=0, max_value=max_conv_layer)):
            if net.shape[-2] == 1:
                break
            elif net.shape[-2] == 2:
                net = self._get_conv1d(net=net, name=f"conv_1d_{r}", kernel_size=2, strides=1)
                break
            else:
                net = self._get_conv1d(net=net, name=f"conv_1d_{r}")

            net = self._wrap_layer(layer=net, name=f"conv_1d_{r}")

        return net

    def __get_conv3d(self, net, name, kernel_size=3, strides=2):
        net = layers.Conv3D(filters=self.hp.Int(f"{name}_1", **self.model_config[TMK.CONV_3D_FILTERS]),
                            kernel_size=kernel_size, padding='valid', activation='relu',
                            kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer,
                            name=f"{name}_1")(net)
        net = layers.Conv3D(filters=self.hp.Int(f"{name}_2", **self.model_config[TMK.CONV_3D_FILTERS]),
                            kernel_size=(1, 1, kernel_size), strides=(1, 1, strides), padding='valid',
                            activation='relu', kernel_initializer=self.kernel_initializer,
                            bias_initializer=self.bias_initializer,
                            name=f"{name}_2")(net)

        return net


class PaperTunerModel1D(PaperTunerModelBase):
    def _get_block(self, net):
        for r in range(self.hp.Int("num_conv1d", min_value=1, max_value=5)):
            if net.shape[-2] == 2:
                net = self._get_conv1d(net=net, name=f"1D_{r}", kernel_size=2, strides=1)
                break
            else:
                net = self._get_conv1d(net=net, name=f"1D_{r}")

        return net
