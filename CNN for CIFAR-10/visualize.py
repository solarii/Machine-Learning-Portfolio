from keras.datasets import cifar10
from keras.utils import np_utils
from utils import visualize_dense_activations, visualize_activations, read_model, get_class_data
from keras.utils import plot_model
from vis.utils import utils
import numpy as np
from keras import backend as K
from matplotlib import pyplot as plt

K.set_image_dim_ordering('th')

# 1. read the model & pretrained weights
model = read_model("saved/final_model.txt", "saved/final_weights.txt")

# 2. print summary and plot model architecture
print(model.summary())
plot_model(model, to_file='model.png')

# 3. get some data to visualize through activations & normalize
(X_train, y_train), (X_test, y_test_orig) = cifar10.load_data()
X_test = X_test.astype('float32')
X_test = X_test / 255.0
y_test = np_utils.to_categorical(y_test_orig)

# 3. visualize
conv_layer_idx = utils.find_layer_idx(model, "conv2d_1")
conv_layer_idx2 = utils.find_layer_idx(model, "conv2d_2")
final_layer_idx = utils.find_layer_idx(model, "dense_2")

airplanes = get_class_data(X_test, y_test_orig, 0)
ships = get_class_data(X_test, y_test_orig, 8)
inputs = X_test[0:50]
input = ships[0:1]

# first conv layer
v2 = visualize_activations(inputs, model, conv_layer_idx, 32, 32)
v1 = visualize_activations(input, model, conv_layer_idx, 32, 32)

# second conv layer
v4 = visualize_activations(inputs, model, conv_layer_idx2, 10, 16)
v3 = visualize_activations(input, model, conv_layer_idx2, 10, 16)

# final layer
v6 = visualize_dense_activations(inputs, model, final_layer_idx)
v5 = visualize_dense_activations(input, model, final_layer_idx)

plt.axis("off")
plt.subplot(3, 2, 1)
plt.title("1. conv. 1 image")
plt.imshow(v1, interpolation='None', cmap="jet")
#plt.imshow(input[0].transpose((1, 2, 0)))

plt.subplot(3, 2, 2)
plt.title("1. conv. 50 images")
plt.imshow(v2, interpolation="None", cmap="jet")

plt.subplot(3, 2, 3)
plt.title("2. conv. 1 image")
plt.imshow(v3, interpolation="nearest", cmap="jet")

plt.subplot(3, 2, 4)
plt.title("2. conv. 50 images")
plt.imshow(v4, interpolation="nearest", cmap="jet")

plt.subplot(3, 2, 5)
plt.title("Final dense 1 image")
plt.imshow(v5, interpolation="nearest", cmap="jet")

plt.subplot(3, 2, 6)
plt.title("Final dense 50 images")
plt.imshow(v6, interpolation="nearest", cmap="jet")

plt.show()
#visualize_dense_activations(input, model, 5, "Final dense layer")
