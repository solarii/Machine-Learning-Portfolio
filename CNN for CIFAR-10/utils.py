import numpy as np
from vis.utils import utils
from keras.models import model_from_json
from keras import backend as K
from math import sqrt


def get_class_data(X, Y, cls):
    idx = Y.flatten()[:] == cls
    return X[idx, :]


def read_model(model_file, weights_file):
    with open(model_file) as myfile:
        json_string = "".join(line.rstrip() for line in myfile)

    model = model_from_json(json_string)
    model.load_weights(weights_file)
    return model


def save_model(model, model_file, weights_file):
    model.save_weights(weights_file)
    json_string = model.to_json()
    with open(model_file, "w") as text_file:
        print("{}".format(json_string), file=text_file)


def get_activations(model, layer, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output])
    activations = get_activations([X_batch, 0])
    return activations


def visualize_activations(input, model, layer_idx, w, h):
    activations = get_activations(model, layer_idx, input)[0]
    filters = np.mean(activations, axis=0)
    vis_images = []
    for i in range(0, filters.shape[0]):
        img = filters[i].reshape((w, h))
        vis_images.append(img.reshape(w, h, 1))

    stitched = utils.stitch_images(vis_images, cols=8)
    rs = stitched.reshape((stitched.shape[0], stitched.shape[1]))
    return rs


def visualize_dense_activations(input, model, layer_idx):
    activations = get_activations(model, layer_idx, input)[0]
    w = int(sqrt(activations.shape[1]))
    filters = np.mean(activations, axis=0)
    filters = filters.reshape((w, w))
    return filters
