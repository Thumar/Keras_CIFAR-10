import tensorflow as tf


def cnn_model():
    input_layer = tf.keras.layers.Input(shape=(32, 32, 3), name="input_layer")
    use_bias = True

    # Conv1
    conv = tf.keras.layers.Conv2D(32,
                                  kernel_size=(3, 3),
                                  padding='same',
                                  use_bias=use_bias,
                                  activation=None)(input_layer)
    bn = tf.keras.layers.BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9)(conv)
    activation = tf.keras.layers.Activation(tf.nn.relu)(bn)

    # Conv2
    conv = tf.keras.layers.Conv2D(32,
                                  kernel_size=(3, 3),
                                  padding='same',
                                  use_bias=use_bias,
                                  activation=None)(activation)
    bn = tf.keras.layers.BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9)(conv)
    activation = tf.keras.layers.Activation(tf.nn.relu)(bn)

    # MaxPooling1
    max_pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(activation)
    dropout = tf.keras.layers.Dropout(0.2)(max_pool)

    # Conv3
    conv = tf.keras.layers.Conv2D(64,
                                  kernel_size=(3, 3),
                                  padding='same',
                                  use_bias=use_bias,
                                  activation=None)(dropout)
    bn = tf.keras.layers.BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9)(conv)
    activation = tf.keras.layers.Activation(tf.nn.relu)(bn)

    # Conv4
    conv = tf.keras.layers.Conv2D(64,
                                  kernel_size=(3, 3),
                                  padding='same',
                                  use_bias=use_bias,
                                  activation=None)(activation)
    bn = tf.keras.layers.BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9)(conv)
    activation = tf.keras.layers.Activation(tf.nn.relu)(bn)

    # MaxPooling2
    max_pool = tf.keras.layers.MaxPooling2D()(activation)
    dropout = tf.keras.layers.Dropout(0.3)(max_pool)

    # Conv5
    conv = tf.keras.layers.Conv2D(128,
                                  kernel_size=(3, 3),
                                  padding='same',
                                  use_bias=use_bias,
                                  activation=None)(dropout)
    bn = tf.keras.layers.BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9)(conv)
    activation = tf.keras.layers.Activation(tf.nn.relu)(bn)
    # Conv6
    conv = tf.keras.layers.Conv2D(128,
                                  kernel_size=(3, 3),
                                  padding='same',
                                  use_bias=use_bias,
                                  activation=None)(activation)
    bn = tf.keras.layers.BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9)(conv)
    activation = tf.keras.layers.Activation(tf.nn.relu)(bn)

    # MaxPooling3
    max_pool = tf.keras.layers.MaxPooling2D()(activation)
    dropout = tf.keras.layers.Dropout(0.4)(max_pool)

    # Dense Layer
    flatten = tf.keras.layers.Flatten()(dropout)

    # Softmax Layer
    output = tf.keras.layers.Dense(10, activation=tf.nn.softmax, name='output')(flatten)

    return tf.keras.Model(inputs=input_layer, outputs=output)
