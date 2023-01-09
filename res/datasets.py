from sklearn import datasets

def get_moons(n_samples=1500, noise=0.1, random_state=42):
    X, y = datasets.make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    return X, y