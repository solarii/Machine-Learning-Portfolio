import numpy
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras import regularizers
from matplotlib import pyplot
from PIL import Image
from utils import save_model
from keras import backend as K

K.set_image_dim_ordering('th')
seed = 8
numpy.random.seed(seed)

# 1. Load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize & Augment
augment = False
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

if augment:
    ImageDataGenerator(featurewise_center=False,
                                featurewise_std_normalization=False,
                                samplewise_center=False,
                                samplewise_std_normalization=False,
                                zca_whitening=False,
                                rotation_range=1,
                                shear_range=0.05,
                                zoom_range=0.05,
                                width_shift_range=0.02,
                                height_shift_range=0.02,
                                rescale=1.0/255,
                                fill_mode="nearest",
                                horizontal_flip=True)
    datagen.fit(X_train)
else:
    X_train = X_train / 255.0
    X_test = X_test / 255.0

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

## training parameters
epochs = 10
alpha = 0.01
weight_decay = alpha / epochs
momentum = 0.9
batch_size = 64

# 2. Create CNN model
model = Sequential()
#model.add(Conv2D(batch_size, (3, 3), input_shape=(3, 32, 32), padding='same', activation='relu', kernel_constraint=maxnorm(3), kernel_regularizer=regularizers.l2(0.01)))
model.add(Conv2D(32, (5, 5), input_shape=(3, 32, 32), padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(3, 2)))

model.add(Conv2D(32, (5,5), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(3, 2)))

#model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
#model.add(MaxPooling2D(pool_size=(3, 3)))

#model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
#model.add(Dropout(0.2))
#model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Flatten())

model.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.02)))
model.add(Dropout(0.3))

model.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.02)))
model.add(Dropout(0.3))

model.add(Dense(num_classes, activation='softmax'))

# 3. Fit
sgd = SGD(lr=alpha, momentum=momentum, decay=weight_decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

#  plot example data from data augmentor
#for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=batch_size):
#    for i in range(0, 9):
#        pyplot.subplot(330 + 1 + i)
#        img = X_batch[i].transpose((1, 2, 0))
#        pyplot.imshow(img)
#    pyplot.show()
#    break
if augment:
    model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size), steps_per_epoch=(len(X_train) / batch_size)*4, epochs=epochs)
else:
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)

# 4. Test data & report performance
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# 5. Save data
save_model(model, "model.txt", "weights.txt")
