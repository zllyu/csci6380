import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from Preprocessing import *
from Model import MLP, Custom
from Train import train


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


def normalization(data):
    # _range = np.max(abs(data))
    # return data / _range
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    filepath = r'../Solar_Flare_Data_Set/flare_data.csv'
    dataset = pd.read_csv(filepath)
    dataset = dataset.drop_duplicates(inplace=False, keep='first')

    X_np, Y_np = transform_data(dataset)
    X_train, X_test, Y_train, Y_test = train_test_split(X_np, Y_np, test_size=0.3, random_state=1)
    X_train = normalization(X_train)
    X_test = normalization(X_test)
    print("X_train size:", X_train.shape, "Y_train size:", Y_train.shape,
          "X_test size:", X_test.shape, "Y_test size:", Y_test.shape)
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    Y_train = torch.tensor(Y_train, dtype=torch.float32).to(device)
    Y_test = torch.tensor(Y_test, dtype=torch.float32).to(device)

    # model = MLP(input_size=24, output_size=3, hidden_list=[512, 256, 128, 64], dropout=0.1).to(device, dtype=torch.float32)

    model = Custom(input_size=24).to(device, dtype=torch.float32)

    batch_size = 64

    train(model, X_train, Y_train, X_test, Y_test,
          epoch=600, lr=0.001, batch_size=batch_size, shuffle=True)


if __name__ == '__main__':
    main()
