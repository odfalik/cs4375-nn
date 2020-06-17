import numpy as np
import pandas as pd
import requests, io
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

data_url                = 'http://utdallas.edu/~rxn170130/cs4375-nn/auto-mpg.csv'
training_testing_split  = 0.7

def getDataset():    
    try:
        # get DF containing raw data
        text_data = io.StringIO(requests.get(data_url).content.decode('utf-8'))

        df = pd.read_csv(text_data, sep=' ')
            

        # split into training and testing datasets
        split_mask = np.random.rand(len(df)) < training_testing_split
        
        training_x_df = df[split_mask][regressors]


        return {
            'training_x_df': df[split_mask][regressors],
            'training_y_df': df[split_mask][[regressand]],
            'testing_x_df': df[~split_mask][regressors],
            'testing_y_df': df[~split_mask][[regressand]]
        }
    except:
        print('Unable to retrieve data')
        quit(code=1)


def main():
    datasets = getDataSets()




if __name == "__main__":
    main()