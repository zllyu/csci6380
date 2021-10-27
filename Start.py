import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from Preprocessing import *
from Model import MLP
from Train import train


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    filepath = r'./dataset/flare_data.csv'
    dataset = pd.read_csv(filepath)
    X_np, Y_np = transform_data(dataset)
    X_train, X_test, Y_train, Y_test = train_test_split(X_np, Y_np, test_size=0.3, random_state=666)
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    Y_train = torch.tensor(Y_train, dtype=torch.float32).to(device)
    Y_test = torch.tensor(Y_test, dtype=torch.float32).to(device)
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

    model = MLP(input_size=24).to(device, dtype=torch.float32)

    batch_size = 32

    train(model, X_train, Y_train, epoch=100, lr=0.1)


if __name__ == '__main__':
    main()
