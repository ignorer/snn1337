import sys
import os
import numpy as np
import theano
import theano.tensor as T
import lasagne


def build_custom_mlp(input_var=None, depth=2, width=[800, 800],
                     output_width=10, drop_input=.2,
                     drop_hidden=.5, shape=(None, 1, 28, 28)):
    # input layer
    network = lasagne.layers.InputLayer(shape=shape,
                                        input_var=input_var,
                                        b=lasagne.init.Uniform())
    if drop_input:
        network = lasagne.layers.dropout(network, p=drop_input)

    # hidden layers
    sigmoid = lasagne.nonlinearities.sigmoid
    for cur_depth in range(depth):
        network = lasagne.layers.DenseLayer(
            network, width[cur_depth], nonlinearity=sigmoid,
            b=lasagne.init.Uniform())
        if drop_hidden:
            network = lasagne.layers.dropout(network, p=drop_hidden)

    # output layer
    network = lasagne.layers.DenseLayer(
        network, output_width, nonlinearity=sigmoid)
    return network


def xor_network():
    """
    Using as example xor_network this function demonstrates how to get weights and biases,
    set weights and biases, check the predictions on the inputs and calculate the error.
    """
    # add inputs and targets to Theano variables
    input_var = T.matrix('inputs')
    target_var = T.matrix('targets')
    inputs = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    targets = np.array([[0.], [1.], [1.], [0.]])

    # create neural network model
    network = build_custom_mlp(input_var, depth=1, width=[2], output_width=1,
                               drop_input=0, drop_hidden=0, shape=(None, 2))

    # set parameters
    params = lasagne.layers.get_all_params(network, trainable=True)
    print("params: ", lasagne.layers.get_all_param_values(network))
    params = [[[20., -20.], [20., -20.]], [-10., 30.], [[20.], [20.]], [-30.]]
    params = [np.array(param) for param in params]
    lasagne.layers.set_all_param_values(network, params)
    params = lasagne.layers.get_all_params(network, trainable=True)
    print("params: ", lasagne.layers.get_all_param_values(network))

    # make prediction
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    pred_fn = theano.function([input_var], [test_prediction])
    pred = pred_fn(inputs)
    print("predictions:", pred)

    # create loss
    test_loss = lasagne.objectives.binary_crossentropy(test_prediction,
                                                       target_var)
    test_loss = test_loss.mean()

    # compile function to check error
    val_fn = theano.function([input_var, target_var], [test_loss],
                             allow_input_downcast=True)
    err = val_fn(inputs, targets)
    print("error:", err)


def load_dataset_digits():
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    samples = load_digits()
    X = samples.data
    y = samples.target
    new_y = [[] for _ in y]
    for i, elem in enumerate(y):
        new_elem = [0 for _ in range(10)]
        new_elem[elem] = 1
        new_y[i] = new_elem
    X_train, X_test, y_train, y_test = train_test_split(
        X, new_y, train_size=0.7)
    return X_train, X_test, y_train, y_test


def saveModelToFile(network, filename="network", shape=(None, 64)):
    """
    Save model to file 'filename' in the following format:
    {width of input layer}\n
    {sequence of weights from each neuron in input layer}\n
    {sequence of weights from bias in input layer}\n
    {width of first hidden layer}\n
    {sequence of weights from each neuron in first hidden layer}\n
    {sequence of weights from bias in first hidden layer}\n
    ...
    {width of first hidden layer}\n
    {sequence of weights from each neuron in last hidden layer}\n
    {sequence of weights from bias in last hidden layer}\n
    {width of output layer}\n
    """
    layers = lasagne.layers.get_all_layers(network)
    params = lasagne.layers.get_all_param_values(network)
    with open(filename, 'w') as f:
        for layer_index, layer in enumerate(layers):
            width = layer.get_output_shape_for(shape)[1]
            f.write(str(width) + "\n")
            if layer_index != len(layers) - 1:
                for neuron_index in range(width):
                    str_params = " ".join(
                        map(str, params[layer_index * 2][neuron_index]))
                    f.write(str_params + "\n")
                str_bias = " ".join(map(str, params[layer_index * 2 + 1]))
                f.write(str_bias + "\n")


def digits_network():
    X_train, X_test, y_train, y_test = load_dataset_digits()

    # prepare Theano variables
    input_var = T.matrix('inputs')
    target_var = T.matrix('targets')

    # create network
    network = build_custom_mlp(
        input_var,
        depth=1,
        width=[100],
        output_width=10,
        drop_input=0,
        drop_hidden=0,
        shape=(None, 64))

    # create loss expresion for training
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.binary_crossentropy(prediction, target_var)
    loss = loss.mean()

    # create update expressions for training
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=0.01, momentum=0.9)

    # create a loss expression for testing
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.binary_crossentropy(test_prediction,
                                                       target_var)
    test_loss = test_loss.mean()

    # compile a function performing a training
    train_fn = theano.function([input_var, target_var], loss, updates=updates,
                               allow_input_downcast=True)

    # compile a second function computing the training loss:
    val_fn = theano.function([input_var, target_var], [test_loss],
                             allow_input_downcast=True)

    num_epochs = 1000
    inputs = np.array(X_train)
    targets = np.array(y_train)
    for epoch in range(num_epochs):
        training_err = train_fn(inputs, targets)
        #print("epoch: {} error: {}".format(epoch, training_err))

    inputs = np.array(X_test)
    targets = np.array(y_test)
    testing_err = val_fn(inputs, targets)
    print("testing err:", testing_err)

    saveModelToFile(network, filename="network_digits", shape=(None, 64))


def main():
    digits_network()


main()
