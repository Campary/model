import keras
from keras.models import Model
from keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D, \
    Activation, Multiply, Dense, GlobalAveragePooling2D, AveragePooling2D, Conv2DTranspose, DepthwiseConv2D, add, \
    SeparableConv2D, ZeroPadding2D, Lambda
from keras.optimizers import Adam
import keras.backend as K
from keras.losses import categorical_crossentropy
import keras.layers as KL
from keras.layers import Layer, InputSpec
from keras import initializers, regularizers, constraints
import tensorflow as tf

def generalized_dice(y_true, y_pred):
    y_true = K.reshape(y_true, shape=(-1, 4))
    y_pred = K.reshape(y_pred, shape=(-1, 4))
    sum_p = K.sum(y_pred, -2)
    sum_r = K.sum(y_true, -2)
    sum_pr = K.sum(y_true * y_pred, -2)
    weights = K.pow(K.square(sum_r) + K.epsilon(), -1)
    generalized_dice = (2 * K.sum(weights * sum_pr)) / (K.sum(weights * (sum_r + sum_p)))
    return generalized_dice

def generalized_dice_loss(y_true, y_pred):
    return 1 - generalized_dice(y_true, y_pred)

def custom_loss(y_true, y_pred):
    return 1 * generalized_dice_loss(y_true, y_pred) + 1 * categorical_crossentropy(y_true, y_pred)

channel_axis = 1 if K.image_data_format() == "channels_first" else 3

def channel_attention(input_xs, reduction_ratio=0.125):

    channel = int(input_xs.shape[channel_axis])
    maxpool_channel = KL.GlobalMaxPooling2D()(input_xs)
    maxpool_channel = KL.Reshape((1, 1, channel))(maxpool_channel)
    avgpool_channel = KL.GlobalAvgPool2D()(input_xs)
    avgpool_channel = KL.Reshape((1, 1, channel))(avgpool_channel)
    Dense_One = KL.Dense(units=int(channel * reduction_ratio), activation='relu', kernel_initializer='he_normal',
                         use_bias=True, bias_initializer='zeros')
    Dense_Two = KL.Dense(units=int(channel), activation='relu', kernel_initializer='he_normal', use_bias=True,
                         bias_initializer='zeros')

    mlp_1_max = Dense_One(maxpool_channel)
    mlp_2_max = Dense_Two(mlp_1_max)
    mlp_2_max = KL.Reshape(target_shape=(1, 1, int(channel)))(mlp_2_max)

    mlp_1_avg = Dense_One(avgpool_channel)
    mlp_2_avg = Dense_Two(mlp_1_avg)
    mlp_2_avg = KL.Reshape(target_shape=(1, 1, int(channel)))(mlp_2_avg)
    channel_attention_feature = KL.Add()([mlp_2_max, mlp_2_avg])
    channel_attention_feature = KL.Activation('sigmoid')(channel_attention_feature)
    return KL.Multiply()([channel_attention_feature, input_xs])


# SAM
def spatial_attention(channel_refined_feature):
    maxpool_spatial = KL.Lambda(lambda x: K.max(x, axis=3, keepdims=True))(channel_refined_feature)
    avgpool_spatial = KL.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(channel_refined_feature)
    max_avg_pool_spatial = KL.Concatenate(axis=3)([maxpool_spatial, avgpool_spatial])
    return KL.Conv2D(filters=1, kernel_size=(3, 3), padding="same", activation='sigmoid',
                     kernel_initializer='he_normal', use_bias=False)(max_avg_pool_spatial)


def cbam(input_xs, reduction_ratio=0.5):
    channel_refined_feature = channel_attention(input_xs, reduction_ratio=reduction_ratio)
    spatial_attention_feature = spatial_attention(channel_refined_feature)
    refined_feature = KL.Multiply()([channel_refined_feature, spatial_attention_feature])
    return KL.Add()([refined_feature, input_xs])

def channel_split(x, num_splits=2):
    if num_splits == 2:
        return tf.split(x, num_or_size_splits=2, axis=-1)
    else:
        raise ValueError('The num_splits is 2')

def _channel_shuffle(x, groups):
    height, width, in_channels = K.int_shape(x)[1:]
    channels_per_group = in_channels // groups
    pre_shape = [-1, height, width, groups, channels_per_group]
    dim = (0, 1, 2, 4, 3)
    later_shape = [-1, height, width, in_channels]
    x = Lambda(lambda z: K.reshape(z, pre_shape))(x)
    x = Lambda(lambda z: K.permute_dimensions(z, dim))(x)
    x = Lambda(lambda z: K.reshape(z, later_shape))(x)
    return x

def FAS(inputs, nb_filter, with_conv_shortcut=False):
    c1_1 = Conv2D(nb_filter, (3, 1), padding="same", kernel_initializer='he_normal')(inputs)
    b1_1 = BatchNormalization(axis=3, gamma_regularizer=regularizers.l2(1e-4), beta_regularizer=regularizers.l2(1e-4))(
        c1_1)
    a1_1 = Activation("relu")(b1_1)
    c1_2 = Conv2D(nb_filter, (1, 3), padding="same", kernel_initializer='he_normal')(a1_1)
    b1_2 = BatchNormalization(axis=3, gamma_regularizer=regularizers.l2(1e-4), beta_regularizer=regularizers.l2(1e-4))(
        c1_2)
    a1_2 = Activation("relu")(b1_2)

    c2_1 = Conv2D(nb_filter, (1, 3), padding="same", kernel_initializer='he_normal')(inputs)
    b2_1 = BatchNormalization(axis=3, gamma_regularizer=regularizers.l2(1e-4), beta_regularizer=regularizers.l2(1e-4))(
        c2_1)
    a2_1 = Activation("relu")(b2_1)
    c2_2 = Conv2D(nb_filter, (3, 1), padding="same")(a2_1)
    b2_2 = BatchNormalization(axis=3, gamma_regularizer=regularizers.l2(1e-4), beta_regularizer=regularizers.l2(1e-4))(
        c2_2)
    a2_2 = Activation("relu")(b2_2)

    concat = concatenate([a1_2, a2_2], axis=-1)
    DeepConv = BatchNormalization()(Conv2D(nb_filter, 3, activation='relu', padding='same', kernel_initializer='he_normal')(concat))
    att1 = cbam(inputs, 0.5)
    if with_conv_shortcut:
        shortcut = BatchNormalization()(
            SeparableConv2D(nb_filter, 3, activation='relu', padding='same', kernel_initializer='he_normal')(att1))
        x = add([DeepConv, shortcut])
        shuffle1 = _channel_shuffle(x, 2)
        return shuffle1
    else:
        x = add([DeepConv, att1])
        shuffle1 = _channel_shuffle(x, 2)
        return shuffle1

def my_model(input_size=(128, 128, 3), classNum=7):
    inputs = Input(input_size)

    conv1 = BatchNormalization()(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs))

    b1 = MaxPooling2D(pool_size=(2, 2))(inputs)
    b2 = MaxPooling2D(pool_size=(4, 4))(inputs)
    b3 = MaxPooling2D(pool_size=(8, 8))(inputs)

    fas1_1 = FAS(conv1, 64)
    fas1_2 = FAS(fas1_1, 64)
    fas1_3 = FAS(fas1_2, 64)
    att1_1 = cbam(fas1_3, 0.5)
    pool1 = MaxPooling2D(pool_size=(2, 2))(fas1_3)

    con2_1 = concatenate([pool1, b1], axis=3)
    y2 = Conv2D(filters=128, kernel_size=1, dilation_rate=1, padding='same', kernel_initializer='he_normal', use_bias=False)(con2_1)
    fas2_1 = FAS(y2, 128, with_conv_shortcut=True)
    fas2_2 = FAS(fas2_1, 128)
    fas2_3 = FAS(fas2_2, 128)
    fas2_4 = FAS(fas2_3, 128)
    att2_1 = cbam(fas2_4, 0.5)
    pool2 = MaxPooling2D(pool_size=(2, 2))(fas2_4)

    con3_1 = concatenate([pool2, b2], axis=3)
    y3 = Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding='same', kernel_initializer='he_normal',use_bias=False)(con3_1)
    fas3_1 = FAS(y3, 256, with_conv_shortcut=True)
    fas3_2 = FAS(fas3_1, 256)
    fas3_3 = FAS(fas3_2, 256)
    fas3_4 = FAS(fas3_3, 256)
    fas3_5 = FAS(fas3_4, 256)
    fas3_6 = FAS(fas3_5, 256)
    att3_1 = cbam(fas3_6, 0.5)
    pool3 = MaxPooling2D(pool_size=(2, 2))(fas3_6)

    con4_1 = concatenate([pool3, b3], axis=3)
    y4 = Conv2D(filters=512, kernel_size=1, dilation_rate=1, padding='same', kernel_initializer='he_normal',use_bias=False)(con4_1)
    fas4_1 = FAS(y4, 512, with_conv_shortcut=True)
    fas4_2 = FAS(fas4_1, 512)
    fas4_3 = FAS(fas4_2, 512)
    att4_1 = cbam(fas4_3, 0.5)
    drop4 = Dropout(0.5)(fas4_3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    fas5_1 = FAS(pool4, 1024, with_conv_shortcut=True)
    drop5_1 = Dropout(0.5)(fas5_1)
    fas5_2 = FAS(drop5_1, 1024)
    drop5_2 = Dropout(0.5)(fas5_2)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5_2))
    con6_1 = concatenate([fas4_3, att4_1, up6], axis=3)
    fas6_1 = FAS(con6_1, 512, with_conv_shortcut=True)
    fas6_2 = FAS(fas6_1, 512, with_conv_shortcut=True)
    drop6_1 = Dropout(0.5)(fas6_2)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop6_1))
    con7_1 = concatenate([fas3_6, att3_1, up7], axis=3)
    fas7_1 = FAS(con7_1, 256, with_conv_shortcut=True)
    fas7_2 = FAS(fas7_1, 256, with_conv_shortcut=True)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(fas7_2))
    con8_1 = concatenate([fas2_4, att2_1, up8], axis=3)
    fas8_1 = FAS(con8_1, 128, with_conv_shortcut=True)
    fas8_2 = FAS(fas8_1, 128, with_conv_shortcut=True)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(fas8_2))
    con9 = concatenate([fas1_3, att1_1, up9], axis=3)
    fas9_1 = FAS(con9, 64, with_conv_shortcut=True)
    fas9_2 = FAS(fas9_1, 64, with_conv_shortcut=True)

    conv10 = Conv2D(classNum, 1, activation='softmax')(fas9_2)

    model = Model(inputs=inputs, outputs=conv10)

    return model