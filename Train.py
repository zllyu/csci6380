import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


def to_string(*kwargs):
    _list = [str(kwargs[0])] + ['{:.6f}'.format(_t) for _t in kwargs[1:]]  # parameters to strings
    total = '\t'.join(_list)  # join these strings to another string
    return total


def evaluate(pred, real):
    total_hit = real == torch.round(pred)
    total_accuracy = total_hit.float().mean()

    mask = real > 0
    hit = real[mask] == torch.round(pred[mask])
    non_zero_acc = hit.float().mean()

    return total_accuracy, non_zero_acc


def train(model, X_train, Y_train, X_test, Y_test, epoch, lr, batch_size, shuffle):
    x = []
    loss_val = []

    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    loss_func = nn.MSELoss()

    train_dataset = data.TensorDataset(X_train, Y_train)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle)

    model.train()
    for t in range(epoch):
        for step, (batch_x, batch_y) in enumerate(train_loader):
            pred = model(batch_x)
            loss = loss_func(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        x.append(t)
        loss_val.append(loss.detach().cpu().numpy())

        print(to_string(t, loss, *evaluate(pred, batch_y)))
        # print(Y_train)
        # print(pred)

    plt.plot(x, loss_val, lw=5)
    plt.title('loss')
    plt.xlabel('steps')
    plt.ylabel('loss')
    plt.show()

    test_pred = model(X_test)
    print(to_string(*evaluate(test_pred, Y_test)))

    # result = torch.cat([Y_train, pred], dim=1)
    # np.savetxt("result.csv", result.detach().cpu().numpy())
