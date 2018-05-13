import numpy as np
import keras
from layers.conv_deform_1d import ConvDeform1D
from layers.ddcconv1d_keras import DDCConv1D


def get_layers_of_type(model: keras.models.Model, layer_type: type):
    return [l for l in model.layers if isinstance(l, layer_type)]


def keras_fetch(model, fetches, input_data):
    inputs = [keras.backend.learning_phase()] + model.inputs
    fn = keras.backend.function(inputs, fetches)
    return fn([0] + [input_data])


def describe_deform_layer(model: keras.models.Sequential, input_data):
    import matplotlib.pyplot as plt

    d_layers = get_layers_of_type(model, ConvDeform1D)
    if not d_layers:
        print('No layers of type ConvDeform1D to describe')
    # Last deformable layer
    layer = d_layers[-1]  # type: ConvDeform1D

    inputs, offsets, deformed = keras_fetch(model, [layer.input, layer.offsets, layer.output], input_data)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    fig.suptitle(layer.name)
    ax1.plot(inputs[0, :, 0], label='Inputs (prev. activations)')
    ax1.plot(deformed[0, :, 0], label='Outputs')
    ax1.grid()
    ax1.legend()
    ax2.plot(offsets[0, :, 0])
    ax2.grid()
    plt.show()


def describe_ddcc_layer(model: keras.models.Model, input_data):
    import matplotlib.pyplot as plt
    d_layers = get_layers_of_type(model, DDCConv1D)  # type: list[DDCConv1D]
    if not d_layers:
        print('No layers of type DDCConv1D to describe')
        return

    for layer in d_layers:
        inputs, offsets, outputs = keras_fetch(model, [layer.input, layer.offsets, layer.output], input_data)
        print('{0}: mean={1:.4f} std={2:.4f}'.format(layer.name, np.mean(offsets), np.std(offsets)))

    # Last deformable layer
    layer = d_layers[-1]
    inputs, offsets, outputs = keras_fetch(model, [layer.input, layer.offsets, layer.output], input_data)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    fig.suptitle(layer.name)
    ax1.plot(inputs[0, :, 0], label='Inputs (prev. activations)')
    ax1.plot(outputs[0, :, 0], label='Outputs')
    ax1.grid()
    ax1.legend()
    if layer.offset_mode == 'SK':
        for i in range(offsets.shape[1]):
            ax2.plot(offsets[:, i])
    ax2.legend()
    ax2.grid()
    plt.show()
