from keras import Input, Model
from keras.layers import Conv2D, MaxPooling2D, concatenate, BatchNormalization, PReLU, SpatialDropout2D, Add, \
    Conv2DTranspose, ReLU, Activation, Permute, ZeroPadding2D, UpSampling2D, Dense, Reshape, Concatenate

class UNET():

    def __init__(self, input_size, nclasses):
        """

        :param input_size: shape of the input image
        :param nclasses: number of classes
        """

        self.im_width = input_size[0]
        self.im_height = input_size[1]
        self.nclasses = nclasses

    def make_conv_block(self, nb_filters, input_tensor, block):
        """

        :param nb_filters: number of filters
        :param input_tensor: input tensor to perform convolution
        :param block: block number (a block has many stages)
        :return:
        """

        def make_stage(input_tensor, stage):
            name = 'conv_{}_{}'.format(block, stage)
            x = Conv2D(nb_filters, (3, 3), activation='relu', padding='same', name=name)(input_tensor)
            name = 'batch_norm_{}_{}'.format(block, stage)
            x = BatchNormalization(name=name)(x)
            x = Activation('relu')(x)
            return x

        # here, a block has 2 stages
        x = make_stage(input_tensor, 1)
        x = make_stage(x, 2)
        return x

    def build(self):
        """
        Build the model for training
        """
        print('. . . . .Building UNET. . . . .')

        # Down-sampling
        inputs = Input(shape=(self.im_width, self.im_height, 3))
        conv1 = self.make_conv_block(32, inputs, 1)     # first block (contains 2 stages)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = self.make_conv_block(64, pool1, 2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = self.make_conv_block(128, pool2, 3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = self.make_conv_block(256, pool3, 4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = self.make_conv_block(512, pool4, 5)
        # End down-sampling

        # Up-sampling
        up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4])  # up-sampling and add skip connection
        conv6 = self.make_conv_block(256, up6, 6)

        up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3])
        conv7 = self.make_conv_block(128, up7, 7)

        up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2])
        conv8 = self.make_conv_block(64, up8, 8)

        up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1])
        conv9 = self.make_conv_block(32, up9, 9)

        conv10 = Conv2D(self.nclasses, (1, 1), name='conv_10_1')(conv9)
        # End up-sampling

        x = Reshape((self.im_width * self.im_height, self.nclasses))(conv10)
        x = Activation('softmax')(x)
        outputs = Reshape((self.im_width, self.im_height, self.nclasses))(x)

        model = Model(inputs=inputs, outputs=outputs)

        print('. . . . .Build Compeleted. . . . .')

        return model