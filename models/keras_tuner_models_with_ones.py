from tensorflow.keras import layers

from models import keras_tuner_model


class KerasTunerModelOnes(keras_tuner_model.KerasTunerModel):
    def __init__(self, name=None, tunable=True):
        super().__init__(name, tunable)

    def model_branch(self, input_, name, i):
        branch = layers.Conv3D(
            filters=self.hp.Int(self.wrap_name(name, f"b{i+2}_1.f"), min_value=16, max_value=128, step=16),
            kernel_size=1,
            padding="same",
            name=self.wrap_name(name, f"b{i+2}_1"))(input_)
        branch = self.wrap_layer(branch, self.wrap_name(name, f"b{i+2}_1"))
        branch = layers.Conv3D(
            filters=self.hp.Int(self.wrap_name(name, f"b{i+2}_2.f"), min_value=16, max_value=128, step=16),
            kernel_size=self.hp.Int(self.wrap_name(name, f"b{i+2}_2.k"), min_value=1, max_value=7, step=2),
            padding="same",
            name=self.wrap_name(name, f"b{i+2}_2"))(branch)
        branch = self.wrap_layer(branch, self.wrap_name(name, f"b{i+2}_2"))

        return branch

    def model_block(self, input_, name):

        branches = []

        b1_name = self.wrap_name(name, "b1")
        branch1 = layers.Conv3D(filters=self.hp.Int(self.wrap_name(name, "b1.f"), min_value=16, max_value=128, step=16),
                                #kernel_size=self.hp.Int(self.wrap_name(name,"b1.л"), min_value=1, max_value=7, step=2),
                                kernel_size=1,
                                padding="same", name=b1_name)(input_)
        branch1 = self.wrap_layer(branch1, b1_name)
        branches.append(branch1)

        for i in range(self.hp.Int(self.wrap_name(name, "num_branches"), 1, 3)):
            branch = self.model_branch(input_, name, i)
            branches.append(branch)

        b_last_name = self.wrap_name(name, "b_last")
        branch_last = layers.MaxPooling3D(pool_size=3,
                                          strides=1,
                                          padding="same",
                                          name=self.wrap_name(b_last_name, 'max_pool'))(input_)
        branch_last = layers.Conv3D(filters=self.hp.Int(self.wrap_name(name, "b_last.f"),
                                                        min_value=16,
                                                        max_value=128,
                                                        step=16),
                                    kernel_size=1,
                                    padding='same',
                                    name=self.wrap_name(b_last_name, 'conv'))(branch_last)
        branch_last = self.wrap_layer(branch_last, b_last_name)
        branches.append(branch_last)

        net = layers.concatenate(branches)

        return net
