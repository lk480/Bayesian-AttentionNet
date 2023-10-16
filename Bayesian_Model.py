import tensorflow_probability as tfp
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Conv2DTranspose,
    concatenate,
    BatchNormalization,
    Activation,
    MaxPooling2D,
    multiply,
)
from keras.models import Model


def variational_unet(
    input_size=(128, 128, 1),
    filters_init=32,
    activation_fn="relu",
    kernel_init="he_normal",
    drop_rate=0.2,
    block_size=7,
):
    # Define Double Convolution Layer
    def conv_layer(filters, inputs, use_spatial_attention=False):
        x = Conv2D(
            filters,
            (3, 3),
            activation=None,
            padding="same",
            kernel_initializer=kernel_init,
        )(inputs)
        x = BatchNormalization()(x)
        x = Activation(activation_fn)(x)
        if use_spatial_attention:
            x = spatial_attention(x)
        x = Conv2D(
            filters,
            (3, 3),
            activation=None,
            padding="same",
            kernel_initializer=kernel_init,
        )(x)
        x = BatchNormalization()(x)
        x = Activation(activation_fn)(x)
        return x

    def variational_layer(filters, inputs):
        x = tfp.layers.Convolution2DFlipout(
            filters, (3, 3), activation=None, padding="same"
        )(inputs)
        x = BatchNormalization()(x)
        x = Activation(activation_fn)(x)
        x = tfp.layers.Convolution2DFlipout(
            filters, (3, 3), activation=None, padding="same"
        )(x)
        x = BatchNormalization()(x)
        x = Activation(activation_fn)(x)
        return x

    def spatial_attention(input_feature):
        """TensorFlow 2 Implementation of Convolutional Block Attention Module - https://arxiv.org/pdf/1807.06521.pdf"""
        kernel_size = 7
        avg_pool = tf.reduce_mean(input_feature, axis=[3], keepdims=True)
        assert avg_pool.shape[-1] == 1
        max_pool = tf.reduce_max(input_feature, axis=[3], keepdims=True)
        assert avg_pool.shape[-1] == 1
        concat = concatenate([avg_pool, max_pool], axis=3)
        assert concat.shape[-1] == 2
        conv_layer = Conv2D(
            filters=1,
            kernel_size=(kernel_size, kernel_size),
            activation=None,
            padding="same",
            kernel_initializer=kernel_init,
            use_bias=False,
        )(concat)
        assert conv_layer.shape[-1] == 1
        sigmoid_output = tf.keras.activations.sigmoid(conv_layer)
        return multiply([input_feature, sigmoid_output])

    inputs = Input(input_size)
    enc_layer1 = conv_layer(filters_init, inputs)
    pool1 = MaxPooling2D((2, 2))(enc_layer1)

    enc_layer2 = conv_layer(2 * filters_init, pool1)
    pool2 = MaxPooling2D((2, 2))(enc_layer2)

    enc_layer3 = conv_layer(4 * filters_init, pool2)
    pool3 = MaxPooling2D((2, 2))(enc_layer3)

    enc_layer4 = conv_layer(8 * filters_init, pool3)
    pool4 = MaxPooling2D((2, 2))(enc_layer4)

    enc_layer5 = conv_layer(16 * filters_init, pool4, use_spatial_attention=True)

    dec_layer4 = concatenate(
        [
            Conv2DTranspose(8 * filters_init, (2, 2), strides=(2, 2), padding="same")(
                enc_layer5
            ),
            enc_layer4,
        ],
        axis=3,
    )
    dec_layer4 = variational_layer(8 * filters_init, dec_layer4)

    dec_layer3 = concatenate(
        [
            Conv2DTranspose(4 * filters_init, (2, 2), strides=(2, 2), padding="same")(
                dec_layer4
            ),
            enc_layer3,
        ],
        axis=3,
    )
    dec_layer3 = variational_layer(4 * filters_init, dec_layer3)

    dec_layer2 = concatenate(
        [
            Conv2DTranspose(2 * filters_init, (2, 2), strides=(2, 2), padding="same")(
                dec_layer3
            ),
            enc_layer2,
        ],
        axis=3,
    )
    dec_layer2 = variational_layer(2 * filters_init, dec_layer2)

    dec_layer1 = concatenate(
        [
            Conv2DTranspose(filters_init, (2, 2), strides=(2, 2), padding="same")(
                dec_layer2
            ),
            enc_layer1,
        ],
        axis=3,
    )
    dec_layer1 = conv_layer(filters_init, dec_layer1)

    out_layer = Conv2D(1, (1, 1), activation="sigmoid")(dec_layer1)

    return Model(inputs=[inputs], outputs=[out_layer])
