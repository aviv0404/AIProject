import numpy as np


def get_diabetes_data():
    data = np.loadtxt("diabetes.csv", delimiter=",", dtype=str)
    x_with_age = data[1:, 1:]
    x_with_age = x_with_age.astype(float)
    x_without_age = np.delete(x_with_age, 18, 1)
    x_without_age = x_without_age.astype(float)
    y = data[1:, 0]
    y = y.astype(float)
    return (normalize_data(x_with_age), normalize_data(x_without_age),
            normalize_data(y))


def get_heart_data():
    # TODO: reformat text to number
    data = np.loadtxt("heart.csv", delimiter=",", dtype=str)
    x = data[1:, :-1]
    y = data[1:, -1]
    return (normalize_data(x), normalize_data(y))


def get_hypothyroid_data():
    # TODO: reformat text to number
    data = np.loadtxt("hypothyroid.csv", delimiter=",", dtype=str)
    x = data[1:, :-1]
    y = data[1:, -1]
    return (normalize_data(x), normalize_data(y))


def get_stroke_data():
    # TODO: reformat text to number
    data = np.loadtxt("stroke.csv", delimiter=",", dtype=str)
    x = data[1:, 1:-1]
    y = data[1:, -1]
    return (normalize_data(x), normalize_data(y))


def normalize_data(x):
    return (x-np.mean(x, axis=0)) / np.var(x, axis=0)


def get_trash_column(x):
    print(np.var(x, axis=0))
    return np.where(np.var(x, axis=0) == 0)[0]


if __name__ == "__main__":
    get_diabetes_data()
