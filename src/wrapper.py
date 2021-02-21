from torch import nn

activation_list = {
    "relu": nn.ReLU(),
    "lrelu": nn.LeakyReLU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "elu": nn.ELU(),
    "selu": nn.SELU(),
    "softmax": nn.Softmax(1),
    "lsoftmax": nn.LogSoftmax(1)
}

def layer_wrapper(n_in, n_out, batch_norm=False, activation='relu', dropout=0.0):
    layers = [nn.Linear(n_in, n_out)]

    if batch_norm:
        layers.append(nn.BatchNorm1d(n_out))

    if activation in activation_list:
        layers.append(activation_list[activation])
    else:
        raise Exception(f"this wrapper supports ({', '.join(activation.keys())})")

    if 0 < dropout <= 1:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)
