import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorflow_model

class_mapping = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_predict = x_test[:44]
y_predict = y_test[:44]

num_class = 10

y_train = tf.keras.utils.to_categorical(y_train, num_class)
y_test = tf.keras.utils.to_categorical(y_test, num_class)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = x_train / 255.0
x_test = x_test / 255.0

data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=90,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

data_gen.fit(x_train)

model = tensorflow_model.cnn_model()

model.summary()

opt_rms = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=opt_rms, metrics=['accuracy'])

batch_size = 128

model = tensorflow_model.cnn_model()

model.summary()

opt_rms = tf.keras.optimizers.RMSprop(lr=0.001, decay=1e-6)
model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=opt_rms, metrics=['accuracy'])

batch_size = 64

log_dir = "training_2"

tbCallback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=50,
                                            write_graph=True, write_grads=True, batch_size=batch_size,
                                            write_images=True)

checkpoint_path = os.path.join(log_dir, "cp.ckpt")

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # Save weights, every 5-epochs.
    period=5)


model.load_weights(checkpoint_path)


def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 75:
        lrate = 0.0005
    elif epoch > 100:
        lrate = 0.0003

    return lrate


model.fit_generator(data_gen.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=x_train.shape[0] // batch_size, epochs=125,
                    verbose=1, validation_data=(x_test, y_test),
                    callbacks=[tf.keras.callbacks.LearningRateScheduler(lr_schedule), cp_callback, tbCallback])

result = model.predict(x_predict / 255.0)

pos = 1
for img, lbl, predict_lbl in zip(x_predict, y_predict, result):
    output = np.argmax(predict_lbl, axis=None)
    plt.subplot(4, 11, pos)
    plt.imshow(img)
    plt.axis('off')
    if output == lbl:
        plt.title(class_mapping[output])
    else:
        plt.title(class_mapping[output] + "/" + class_mapping[lbl[0]], color='#ff0000')
    pos += 1

plt.show()

scores = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)

print('\nTest accuracy: %.3f loss: %.3f' % (scores[1] * 100, scores[0]))
