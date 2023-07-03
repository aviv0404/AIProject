import numpy as np

def get_diabetes_data():
    data = np.loadtxt("diabetes.csv", delimiter=",", dtype=str)
    x_with_age = data[1:, 1:]
    x_without_age = np.delete(x_with_age, 18, 1)
    print(x_without_age[0])
    print(x_with_age[0])
    y = data[0, 1:]
    return (x_with_age, x_without_age, y)

def get_heart_data():
    # TODO: reformat text to number
    data = np.loadtxt("heart.csv", delimiter=",", dtype=str)
    x = data[1:, :-1]
    y = data[1:, -1]
    return (x, y)

def get_hypothyroid_data():
    # TODO: reformat text to number
    data = np.loadtxt("hypothyroid.csv", delimiter=",", dtype=str)
    x = data[1:, :-1]
    y = data[1:, -1]
    return (x, y)

def get_stroke_data():
    # TODO: reformat text to number
    data = np.loadtxt("stroke.csv", delimiter=",", dtype=str)
    x = data[1:, 1:-1]
    y = data[1:, -1]
    return (x, y)

# TODO: alot

if __name__ == "__main__":
    get_diabetes_data()
