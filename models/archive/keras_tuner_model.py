import keras_tuner
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# import config
D3_SIZE = [3, 3]
NUMBER_OF_CLASSES_TO_TRAIN = 8
OUTPUT_SIGNATURE_X_FEATURES = 92


class KerasTunerModel(keras_tuner.HyperModel):

    def __init__(self, conf: dict, name=None, tunable=True):
        super().__init__(name, tunable)
        self.config = conf

    def declare_hyperparameters(self, hp):
        pass

    @staticmethod
    def wrap_name(layer_name, name):
        return layer_name + "." + name

    def get_activations(self, name):
        return self.hp.Choice(self.wrap_name(name, "activation"), self.config["ACTIVATION"])

    def get_optimizer(self):
        optimizer = self.hp.Choice('optimizer', [key for key in self.config["OPTIMIZER"].keys()])
        lr = self.hp.Float("lr", **self.config["LEARNING_RATE"])
        return self.config[optimizer](learning_rate=lr)

    def wrap_layer(self, layer, name):
        order_activation_batch_norm = self.hp.Choice(self.wrap_name(name, "mode"), ['without_batch_norm',
                                                                                    'batch_norm_activation',
                                                                                    'activation_batch_norm'])

        batch_name = self.wrap_name(name, 'batch_norm')
        activation_name = self.wrap_name(name, 'activation')
        dropout_name = self.wrap_name(name, 'dropout')

        if order_activation_batch_norm == "batch_norm_activation":
            layer = layers.BatchNormalization(name=batch_name)(layer)   # no batch_normalization

        layer = layers.Activation(self.get_activations(name), name=activation_name)(layer)

        if order_activation_batch_norm == "activation_batch_norm":
            layer = layers.BatchNormalization(name=batch_name)(layer)   # batch_normalization than activation

        if self.hp.Boolean(self.wrap_name(name, "dr")):
            layer = layers.Dropout(self.hp.Float(self.wrap_name(name, "dr_val"), **self.config["DROPOUT"]),
                                   name=dropout_name)(layer)

        return layer

    def model_branch(self, input_, name, i,):
        branch = input_
        for idx in range(1, 3):
            branch = layers.Conv3D(
                filters=self.hp.Int(self.wrap_name(name, f"b{i+1}_{idx}.f"), **self.config["BLOCK_FILTERS"]),
                kernel_size=self.hp.Int(self.wrap_name(name, f"b{i+1}_{idx}.k"), **self.config["BLOCK_KERNEL_SIZE"]),
                padding="same",
                name=self.wrap_name(name, f"b{i + 1}_{idx}"))(branch)
            branch = self.wrap_layer(branch, self.wrap_name(name, f"b{i + 1}_{idx}"))

        return branch

    def model_block(self, input_, name):

        branches = []

        b1_name = self.wrap_name(name, "b1")
        branch1 = layers.Conv3D(filters=self.hp.Int(self.wrap_name(name,"b1.f"), min_value=16, max_value=128, step=16),
                                #kernel_size=self.hp.Int(self.wrap_name(name,"b1.л"), min_value=1, max_value=7, step=2),
                                kernel_size=1,
                                padding="same", name=b1_name)(input_)
        branch1 = self.wrap_layer(branch1, b1_name)
        branches.append(branch1)

        for i in range(self.hp.Int(self.wrap_name(name, "num_branches"), 1, 3)):
            branch = self.model_branch(input_, name, i)
            branches.append(branch)

        b_last_name = self.wrap_name(name, "b_last")
        branch_last = layers.MaxPooling3D(pool_size=3, strides=1, padding="same", name=self.wrap_name(b_last_name, 'max_pool'))(input_)
        branch_last = layers.Conv3D(filters=self.hp.Int(self.wrap_name(name,"b_last.f"), min_value=16, max_value=128, step=16),
                                kernel_size=self.hp.Int(self.wrap_name(name,"b_last.k"), min_value=1, max_value=7, step=2),
                                padding='same',
                                name=self.wrap_name(b_last_name, 'conv'))(branch_last)
        branch_last = self.wrap_layer(branch_last, b_last_name)
        branches.append(branch_last)

        #net = layers.concatenate([branch1, branch2, branch3, branch_last])
        net = layers.concatenate(branches)

        return net

    def model(self):
        input_ = layers.Input(
            shape=(D3_SIZE[0],
                   D3_SIZE[1],
                   OUTPUT_SIGNATURE_X_FEATURES),
            name="title"
        )

        net = tf.expand_dims(input_, axis=-1)
        for bl in range(self.hp.Int("num_blocks", 0, 2)):
            net = self.model_block(net, f"bl{bl}")

        net = layers.Flatten()(net)

        for fc in range(self.hp.Int("num_fc", 0, 2)):
            net = tf.keras.layers.Dense(self.hp.Int(f'fc_{fc}', min_value=8, max_value=32, step=4),
                                        activation=self.get_activations(f'fc_{fc}'))(net)

            if self.hp.Boolean(f"fc_{fc}_dropout"):
                net = layers.Dropout(self.hp.Float(f"fc_{fc}_dropout_value", min_value=0.1, max_value=0.9, step=0.1))(net)

        activation = 'sigmoid'
        number = 1
        if NUMBER_OF_CLASSES_TO_TRAIN > 2:
            activation = None
            number = NUMBER_OF_CLASSES_TO_TRAIN

        result = tf.keras.layers.Dense(number, activation=activation)(net)

        model = tf.keras.Model(
            inputs=[input_],
            outputs=[result]
        )

        return model

    def build(self, hp):
        self.hp = hp
        model = self.model()

        if NUMBER_OF_CLASSES_TO_TRAIN == 2:
            loss = keras.losses.BinaryCrossentropy(),
            metrics = keras.metrics.BinaryAccuracy(name='accuracy')
        else:
            loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics = keras.metrics.SparseCategoricalAccuracy()

        print('metrics', metrics)

        model.compile(
            optimizer=self.get_optimizer(),
            loss=loss,
            metrics=metrics
        )

        return model

    def fit(self, hp, model, class_weight=None, *args, **kwargs):
        model.summary()

        #print(class_weight)
        #print(', \n'.join(['{}={!r}'.format(k, v) for k, v in kwargs.items()]))

        class_weights = None
        if hp.Boolean('class_weights'):
            class_weights = class_weight

        return model.fit(
            class_weight=class_weights,
            **kwargs
        )


if __name__ == "__main__":
    pass
