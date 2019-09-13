import tensorflow as tf

INITIALIZER = tf.random_normal_initializer(0., 0.02)
OUTPUT_CHANNELS = 3


def downsample(filters, size, layer, apply_batchnorm=True):
    """
    Un bloque convolucional para el encoder del modelo pix2pix.
        Args:
        - filters: La cantidad de filtros que tendra la capa 
                   convolucional.
        - size: El tamaño del kernel.
        - layer: La capa anterior a la que se conectara este bloque.
        - apply_batchnorm: True o False, si aplicamos o no el
                           BatchNormalization.
        
        Returns: 
            layer: El bloque conectado, Una Conv2D y BatchNormalization 
                   si es que se aplicó.
    """
    layer = tf.keras.layers.Conv2D(
        filters,
        size,
        strides=2,
        padding='same',
        kernel_initializer=INITIALIZER,
        use_bias=False,
        activation=tf.nn.leaky_relu
    )(layer)
    if apply_batchnorm:
        layer = tf.keras.layers.BatchNormalization()(layer)
    
    return layer

def upsample(filters, size, layer, apply_dropout=False):
    """
    Un bloque deconvolucional para el decoder del modelo pix2pix.
        Args:
        - filters: La cantidad de filtros que tendra la capa 
                   convolucional.
        - size: El tamaño del kernel.
        - layer: La capa anterior a la que se conectara este bloque.
        - apply_dropout: True o False, si aplicamos o no el
                           Dropout.
        
        Returns: 
            layer: El bloque conectado, Una Conv2D, BatchNormalization
            y Dropout si es que se aplicó.
    """
    layer = tf.keras.layers.Conv2DTranspose(
        filters,
        size,
        strides=2,
        padding='same',
        kernel_initializer=INITIALIZER,
        use_bias=False,
        activation=tf.nn.relu
    )(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    if apply_dropout:
        layer = tf.keras.layers.Dropout(0.5)(layer)
    return layer

def Generator(name):
    """
    Este es el modelo generador pix2pix basado en el paper https://arxiv.org/pdf/1611.07004.pdf.
    Este modelo Generará una imagen a partir de otra según el problema que quieres resolver.
        Returns:
            model: retorna un modelo de la libreria tf.keras.        
    """
    down_stack_parameters = [
        [64, 4, False], # (bs, 128, 128, 64)
        [128, 4, True], # (bs, 64, 64, 128)
        [256, 4, True], # (bs, 32, 32, 256)
        [512, 4, True], # (bs, 16, 16, 512)
        [512, 4, True], # (bs, 8, 8, 512)
        [512, 4, True], # (bs, 4, 4, 512)
        [512, 4, True], # (bs, 2, 2, 512)
        [512, 4, True], # (bs, 1, 1, 512)
    ]

    up_stack_parameters = [
        [512, 4, True], # (bs, 2, 2, 1024)
        [512, 4, True], # (bs, 4, 4, 1024)
        [512, 4, True], # (bs, 8, 8, 1024)
        [512, 4, False], # (bs, 16, 16, 1024)
        [256, 4, False], # (bs, 32, 32, 512)
        [128, 4, False], # (bs, 64, 64, 256)
        [64, 4, False], # (bs, 128, 128, 128)
    ]

    inputs = tf.keras.layers.Input(shape=[None, None, 3])
    x = inputs

    # Downsampling through the model
    skips = []
    for filters, size, app_batchnorm in down_stack_parameters:
        x = downsample(filters, size, x, app_batchnorm)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up_stack, skip in zip(up_stack_parameters, skips):
        filters, size, apply_dropout = up_stack
        x = upsample(filters, size, x, apply_dropout)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = tf.keras.layers.Conv2DTranspose(
        OUTPUT_CHANNELS,
        4,
        strides=2,
        padding='same',
        kernel_initializer=INITIALIZER,
        activation=tf.nn.tanh
    )(x) # (bs, 256, 256, 3)(x)
    
    return tf.keras.Model(inputs=inputs, outputs=x, name=name)


def Discriminator(name):
    """
    Un modelo Discriminador que se encargara de diferenciar una imagen generada de una real.
    Este modelo sera parte de la evaluacion de error para el entrenamiento del generador.
    El modelo tambien esta basado el el paper https://arxiv.org/pdf/1611.07004.pdf
    """

    input_image = tf.keras.layers.Input(shape=[None, None, 3], name='input_image')
    target_image = tf.keras.layers.Input(shape=[None, None, 3], name='target_image')

    x = tf.keras.layers.concatenate([input_image, target_image])  # (bs, 256, 256, channels*2)
    x = downsample( 64, 4, x, False)       # (bs, 128, 128, 64)
    x = downsample(128, 4, x)              # (bs, 64, 64, 128)
    x = downsample(256, 4, x)              # (bs, 32, 32, 256)
    x = tf.keras.layers.ZeroPadding2D()(x) # (bs, 34, 34, 256)
    x = downsample(512, 4, x)              # (bs, 31, 31, 512)
    x = tf.keras.layers.ZeroPadding2D()(x) # (bs, 33, 33, 512)

    x = tf.keras.layers.Conv2D(
        1, 4, strides=1,
        kernel_initializer=INITIALIZER
    )(x)  # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=[input_image, target_image], outputs=x, name=name)
