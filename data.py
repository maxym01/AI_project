import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def get_data():

    raw_dataset = pd.read_csv('CarPrice_Assignment.csv')

    # remove entries with missing values
    dataset = raw_dataset.dropna()
    # from sklearn import preprocessing
    # normalized_features = preprocessing.StandardScaler().fit_transform(dataset)
    # dataset = pd.DataFrame(data=normalized_features, columns=column_names)
    return dataset


def inspect_data(dataset):
    print('Dataset shape:')
    print(dataset.shape)

    print('Tail:')
    print(dataset.tail())

    print('Statistics:')
    print(dataset.describe().transpose())

    sns.pairplot(dataset[['enginesize', 'wheelbase', 'price', 'curbweight']], diag_kind='kde')
    plt.show()


def split_data(dataset):
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    return train_dataset, test_dataset
