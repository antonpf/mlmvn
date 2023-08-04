import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn import datasets


def get_moons(n_samples=1500, noise=0.1, random_state=42):
    X, y = datasets.make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    return X, y

def get_glass(use_local_data = False):
    if use_local_data:
        column_names = ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]
        X_df = pd.read_csv(
            "/media/anton/Daten/Dev/Python/CVNNMVN/examples/glass classification/glass_setX.csv",
            names=column_names,
        )
        # X_df = pd.read_csv('/Users/anton/Documents/Dev/Python/CVNNMVN/examples/glass classification/glass_setX.csv', names=column_names)
        column_names = ["Type of glass"]
        y_df = pd.read_csv(
            "/media/anton/Daten/Dev/Python/CVNNMVN/examples/glass classification/glass_setY.csv",
            names=column_names,
        )
        # y_df = pd.read_csv('/Users/anton/Documents/Dev/Python/CVNNMVN/examples/glass classification/glass_setY.csv', names=column_names)
        X, y = X_df.to_numpy(), y_df.to_numpy()
    else:
        column_names = [
            "ID",
            "RI",
            "Na",
            "Mg",
            "Al",
            "Si",
            "K",
            "Ca",
            "Ba",
            "Fe",
            "Type of glass",
        ]
        df = pd.read_csv(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data",
            names=column_names,
        )
        df.drop(["ID"], axis=1, inplace=True)

        data = df.to_numpy()
        X, y = data[:, :-1], data[:, -1]
        y = y - 1
        # label encode the target variable to have the classes 0 and 1
        y = LabelEncoder().fit_transform(y)
        X, y = shuffle(X, y, random_state=0)
        
    return X, y
