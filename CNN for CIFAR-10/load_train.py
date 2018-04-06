from keras.datasets import cifar10
from keras.utils import np_utils
from utils import visualize_dense_activations, visualize_activations, read_model, get_class_data, save_model
from keras.utils import plot_model
from keras.optimizers import SGD
from vis.utils import utils
import numpy as np
from keras import backend as K
from matplotlib import pyplot as plt

K.set_image_dim_ordering('th')

# 1. read the model & pretrained weights
model = read_model("model.txt", "weights.txt")

# 2. print summary and plot model architecture
print(model.summary())

# 3. get some data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
augment = False
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
if augment:
    datagen = ImageDataGenerator(featurewise_center=False,
                                featurewise_std_normalization=False,
                                samplewise_center=False,
                                samplewise_std_normalization=False,
                                zca_whitening=False,
                                rotation_range=3,
                                shear_range=0.1,
                                zoom_range=0.1,
                                width_shift_range=0.05,
                                height_shift_range=0.05,
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

# 4. compile
## training parameters
epochs = 150
alpha = 0.01
weight_decay = alpha / epochs
momentum = 0.9
batch_size = 64

sgd = SGD(lr=alpha, momentum=momentum, decay=weight_decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# first conv layer
if augment:
    model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size), steps_per_epoch=(len(X_train) / batch_size)*4, epochs=epochs)
else:
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)

# 4. Test data & report performance
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# 5. Save data
save_model(model, "final_model.txt", "final_weights.txt")
