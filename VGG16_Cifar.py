from tensorflow.keras.datasets import cifar10
from tensorflow import keras
from tensorflow.keras import optimizers, initializers, regularizers, metrics
from tensorflow_core.python.keras.backend import dropout
from tensorflow_core.python.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from tensorflow_core.python.keras import Input, Model
from tensorflow_core.python.keras.layers import Conv2D, GlobalAveragePooling2D, BatchNormalization,UpSampling2D
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from tensorflow_core.python.layers.core import flatten
import matplotlib.pyplot as plt
EPOCHS = 30

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print('the data format: ', X_train.shape)
print("train data：%2.0f, test data：%2.0f" % (X_train.shape[0], X_test.shape[0]))

X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255


# X_train[0][:5, :, 1] # the picture matrix shows the first 5 rows of the R matrix
#
# fig, ax = plt.subplots()
# ax.imshow(X_train[1])
#
# fig = plt.figure(figsize = (20, 5))
# for i in range(20):
#     ax = fig.add_subplot(2, 10, i + 1, xticks = [], yticks = [])
#     ax.imshow(X_train[i])
# y_train[:20].reshape(2, 10)

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

inputs = Input(shape=(3, 32, 32))

x = UpSampling2D(size=(2,2))(inputs)

# block1
x = Conv2D(64, (3, 3), padding='same', name='block1_conv/1')(inputs)
x = BatchNormalization(axis=1)(x)
x = Activation("relu")(x)
x = Conv2D(64, (3, 3), padding='same', name='block1_conv/2')(x)
x = BatchNormalization(axis=1)(x)
x = Activation("relu")(x)
# x = DropBlock(block_size=7, keep_prob=0.9)(x)  # drop
x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
# x = Dropout(0.20)(x)

# block2
x = Conv2D(128, (3, 3), padding='same', name='block2_conv1')(x)
x = BatchNormalization(axis=1)(x)
x = Activation("relu")(x)
x = Conv2D(128, (3, 3), padding='same', name='block2_conv2')(x)
x = BatchNormalization(axis=1)(x)
x = Activation("relu")(x)
# x = DropBlock(block_size=7, keep_prob=0.9)(x)  # drop
x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
x = Dropout(0.20)(x)

# block3
x = Conv2D(256, (3, 3), padding='same', name='block3_conv1')(x)
x = BatchNormalization(axis=1)(x)
x = Activation("relu")(x)
x = Conv2D(256, (3, 3), padding='same', name='block3_conv2')(x)
x = BatchNormalization(axis=1)(x)
x = Activation("relu")(x)
x = Conv2D(256, (3, 3), padding='same', name='block3_conv3')(x)
x = BatchNormalization(axis=1)(x)
x = Activation("relu")(x)
# x = DropBlock(block_size=7, keep_prob=0.9)(x)  # drop
x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

# block4
x = Conv2D(512, (3, 3), padding='same', name='block4_conv1')(x)
x = BatchNormalization(axis=1)(x)
x = Activation("relu")(x)
x = Conv2D(512, (3, 3), padding='same', name='block4_conv2')(x)
x = BatchNormalization(axis=1)(x)
x = Activation("relu")(x)
x = Conv2D(512, (3, 3), padding='same', name='block4_conv3')(x)
x = BatchNormalization(axis=1)(x)
x = Activation("relu")(x)
# x = DropBlock(block_size=4, keep_prob=0.7)(x)  # drop
x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

# block5
x = Conv2D(512, (3, 3), padding='same', name='block5_conv1')(x)
x = BatchNormalization(axis=1)(x)
x = Activation("relu")(x)
x = Conv2D(512, (3, 3), padding='same', name='block5_conv2')(x)
x = BatchNormalization(axis=1)(x)
x = Activation("relu")(x)
x = Conv2D(512, (3, 3), padding='same', name='block5_conv3')(x)
x = BatchNormalization(axis=1)(x)
x = Activation("relu")(x)
# x = DropBlock(block_size=7, keep_prob=0.9)(x)  # drop
x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

# Classification block
x = GlobalAveragePooling2D(data_format='channels_first', name='fc1')(x)
x = Flatten(name='Flatten')(x)

x = Dropout(0.2)(x)
x = Dense(1024, activation='relu', name='fc2')(x)
# x = BatchNormalization(axis=1)(x)
# x = Dense(1024, activation='relu', name='fc3')(x)
# x = Dropout(0.2)(x)
predictions = Dense(10, activation='softmax', name='fc4')(x)


cifar = Model(inputs=inputs, outputs=predictions)
optimizer = optimizers.SGD(learning_rate=0.001,momentum=0.9)
cifar.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])

show = cifar.summary()
print(show)
#
# checkpoint = ModelCheckpoint(filepath='My_VGG_weight.hdf5',
#                              monitor='loss',
#                              mode='min',
#                              save_best_only=True)

cifar.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['acc'])

history = cifar.fit(X_train,
                    y_train,
                    epochs=EPOCHS,
                    validation_data=(X_test, y_test)

                    )

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'ro', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Loss')
plt.legend()

plt.show()

cifar.save('cifar04.h5')

# callbacks = [checkpoint]


test_loss, test_acc = cifar.evaluate(X_test,y_test)