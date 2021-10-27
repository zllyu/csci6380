import torch
import torch.nn as nn


def to_string(*kwargs):
    _list = [str(kwargs[0])] + ['{:.6f}'.format(_t) for _t in kwargs[1:]]  # parameters to strings
    total = '\t'.join(_list)  # join these strings to another string
    return total


def evaluate(pred, real):
    total_hit = real == torch.round(pred)
    total_accuracy = total_hit.float().mean()
    return total_accuracy


def train(model, X_train, Y_train, epoch, lr):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_func = nn.MSELoss()

    for t in range(epoch):
        pred = model(X_train)
        loss = loss_func(pred, Y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(to_string('Accuracy: ', evaluate(pred, Y_train)))
