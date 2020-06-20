import numpy as np
import pandas as pd
import requests, io
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from ann import ANN

data_url                = 'http://utdallas.edu/~rxn170130/4375-data/semeion.data'
test_proportion         = 0.3
input_dim               = 256
output_dim              = 10

# Fetch datasets, preprocess, and split into training and testing x/y datasets
def getDatasets():
    # get data_m containing raw data
    text_data = io.StringIO(requests.get(data_url).content.decode('utf-8'))

    data_m = pd.read_csv(text_data, delim_whitespace=True, header=None).to_numpy()

    # Split into training and testing datasets
    train_m, test_m = train_test_split(data_m, test_size=test_proportion)

    # TODO: verify these splits
    train_x_m = train_m[:, :input_dim]
    train_y_m = train_m[:, input_dim:]
    test_x_m  = test_m[:, :input_dim]
    test_y_m  = test_m[:, input_dim:]

    # print(f'categorized as {np.nonzero(train_y_m[0])[0][0]}')
    # plt.imshow(np.reshape(train_x_m[0], (16,16)), interpolation='nearest')
    # plt.show()

    return {
        'train_x_m': train_x_m,
        'train_y_m': train_y_m,
        'test_x_m': test_x_m,
        'test_y_m': test_y_m
    }


def main():
    datasets = getDatasets()

    # Initialize NN instance
    nn = ANN(
        dims=(256, 3, 3, 10),
        activation='sigmoid'
    )

    nn.train(
        x_m=datasets['train_x_m'],
        y_m=datasets['train_y_m'],
        learning_rate=0.0008,
        max_batch_size=5,
        momentum_factor=0.2
    )

    # nn.test(
    #     x_m=datasets['test_x_m'],
    #     y_m=datasets['test_y_m']
    # )




if __name__ == "__main__":
    main()