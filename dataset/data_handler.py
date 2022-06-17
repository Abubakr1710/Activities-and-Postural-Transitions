import pandas as pd
import numpy as np
import os
np.random.seed(0)


def train_test_split(rawDataFolder='dataset/RawData', train_size=0.7):

    # read file containing infos about experimentID, userID, label(activity), start and end of sequence
    col_names = ['exp', 'user', 'label', 'start', 'end']
    file_infos = pd.read_csv(os.path.join(
        rawDataFolder, 'labels.txt'), sep=' ', header=None, names=col_names)

    # UserID for train and test sets
    users = file_infos['user'].unique()
    np.random.shuffle(users)
    train_users = users[:int(len(users)*0.7)]
    test_users = users[int(len(users)*0.7):]

    # create Train and Test folders
    os.mkdir('train')
    os.mkdir('test')

    pass


train_test_split()
