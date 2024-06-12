import abc

from keras import layers, Model

from models.kt_hypermodel_base import KtModelBase
from models.model_randomness import get_dropout

from configuration.keys import TunerModelKeys as TMK


class InceptionTunerModelBase(KtModelBase):
    def _model(self, input_layer: layers.Input, net: layers.Reshape):
        for bl in range(self.hp.Int("num_blocks", **self.model_config[TMK.NUM_BLOCK])):
            net = self.__model_block(input_=net, name=f"bl{bl}")

        net = layers.Flatten()(net)

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

        result = layers.Dense(number, activation=activation, name="output", kernel_initializer=self.kernel_initializer,
                              bias_initializer=self.bias_initializer, )(net)

        model = Model(
            inputs=[input_layer],
            outputs=[result]
        )

        return model

    def __model_block(self, input_, name):
        branches = []

        b1_name = self._wrap_name(name, "b1")
        b1_f_name = self._wrap_name(b1_name, "f")
        branch1 = self._get_conv_layer(input_=input_, name=b1_name, f_name=b1_f_name, k_name="", kernel_size=1)
        branch1 = self._wrap_layer(branch1, b1_name)
        branches.append(branch1)

        for i in range(self.hp.Int(self._wrap_name(name, "num_branches"), **self.model_config[TMK.NUM_BRANCHES])):
            branch = self.__model_branch(input_, name, i)
            branches.append(branch)

        b_last_name = self._wrap_name(name, "b_last")
        b_last_f_name = self._wrap_name(b_last_name, "f")
        b_last_k_name = self._wrap_name(b_last_name, "k")
        branch_last = self._get_max_pool(input_=input_, name=b_last_name)
        with_ones = self.model_config[TMK.WITH_ONES]
        branch_last = self._get_conv_layer(input_=branch_last, name=b_last_name, f_name=b_last_f_name,
                                           k_name=b_last_k_name, kernel_size=1 if with_ones else None)
        branch_last = self._wrap_layer(branch_last, b_last_name)
        branches.append(branch_last)

        net = layers.concatenate(branches)

        return net

    def __model_branch(self, input_, name, idx):
        branch = input_
        for idx_sub in range(1, 3):
            b_name = self._wrap_name(name, f"b{idx + 1}_{idx_sub}")
            f_name = self._wrap_name(b_name, "f")
            k_name = self._wrap_name(b_name, "k")
            with_ones = self.model_config[TMK.WITH_ONES]
            branch = self._get_conv_layer(input_=branch, name=b_name, f_name=f_name, k_name=k_name,
                                          kernel_size=1 if with_ones & idx_sub == 1 else None)
            branch = self._wrap_layer(branch, b_name)

        return branch

    @abc.abstractmethod
    def _get_conv_layer(self, input_, name, f_name, k_name, kernel_size=None):
        pass

    @abc.abstractmethod
    def _get_max_pool(self, input_, name):
        pass


class InceptionTunerModel3D(InceptionTunerModelBase):
    def _get_conv_layer(self, input_, name, f_name, k_name, kernel_size=None):
        return layers.Conv3D(
            filters=self.hp.Int(f_name, **self.model_config[TMK.BLOCK_FILTERS]),
            kernel_size=self.hp.Int(k_name, **self.model_config[TMK.BLOCK_KERNEL_SIZE]) if kernel_size is None
            else kernel_size, kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer,
            padding="same",
            name=name)(input_)

    def _get_max_pool(self, input_, name):
        return layers.MaxPooling3D(pool_size=3, strides=1, padding="same",
                                   name=self._wrap_name(name, 'max_pool'))(input_)


class InceptionTunerModel1D(InceptionTunerModelBase):
    def _get_conv_layer(self, input_, name, f_name, k_name, kernel_size=None):
        return layers.Conv1D(
            filters=self.hp.Int(f_name, **self.model_config[TMK.BLOCK_FILTERS]),
            kernel_size=self.hp.Int(k_name, **self.model_config[TMK.BLOCK_KERNEL_SIZE]) if kernel_size is None
            else kernel_size, kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer,
            padding="same",
            name=name)(input_)

    def _get_max_pool(self, input_, name):
        return layers.MaxPooling1D(pool_size=3, strides=1, padding="same",
                                   name=self._wrap_name(name, 'max_pool'))(input_)
