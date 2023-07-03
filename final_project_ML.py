import numpy as np


def get_diabetes_data():
    # get the data, remove top row of column names
    data = np.loadtxt("diabetes.csv", delimiter=",", dtype=str)
    x = data[1:, 1:]
    x = x.astype(float)

    # deleting unecessary data with variance = 0
    x = np.delete(x, get_trash_column(x), 1)

    # y is first column
    y = data[1:, 0]
    y = y.astype(float)

    # return normalized data
    return (normalize_data(x), normalize_data(y))


def get_heart_data():
    # get the data
    data = np.loadtxt("heart.csv", delimiter=",", dtype=str)
    x = data[1:, :-1]
    y = data[1:, -1]

    # reformat text to numbers
    column_sets = [set(column) for column in np.transpose(x)]

    # let's see which columns need reformatting:
    for column in range(len(column_sets)):
        print(str(column) + ": " + str(column_sets[column]) + "\n")

    # we need to work on columns: 1,2,6,8,10.
    #  we'll assign a number to each unique entry
    # for examle in column 10 we have {'Flat', 'Up', 'Down'}
    # so we'll convert it into: {0,1,2}

    # column 1: Sex
    condition_female = x[:, 1] == "F"
    # Change 'F' to 0 and 'M' to 1 in column 1
    x[condition_female, 1] = 0
    x[~condition_female, 1] = 1

    # column 2: Chest Pain
    chestPainMapping = {"NAP": 0, "ATA": 1, "TA": 2, "ASY": 3}
    v_mapping = np.vectorize(lambda x: chestPainMapping[x])
    x[:, 2] = v_mapping(x[:, 2])

    # column 6: Resting ECG
    ecgMapping = {"Normal": 0, "LVH": 1, "ST": 2}
    v_mapping = np.vectorize(lambda x: ecgMapping[x])
    x[:, 6] = v_mapping(x[:, 6])

    # column 8: Exercise Angina
    condition_angina = x[:, 8] == "N"
    # Change 'N' to 0 and 'Y' to 1 in column 8
    x[condition_angina, 8] = 0
    x[~condition_angina, 8] = 1

    # column 10: ST Slope
    slopeMapping = {"Up": 1, "Flat": 0, "Down": -1}
    v_mapping = np.vectorize(lambda x: slopeMapping[x])
    x[:, 10] = v_mapping(x[:, 10])
    
    x = x.astype(float)
    y = y.astype(float)

    return (normalize_data(x), normalize_data(y))


def get_hypothyroid_data():
    # TODO: reformat text to number
    data = np.loadtxt("hypothyroid.csv", delimiter=",", dtype=str)
    x = data[1:, :-1]
    y = data[1:, -1]
    # remove unrelevent column
    x = np.delete(x, 27, 1)
    # Remove the rows that contain '?'
    conditionUndifined = np.any(x == '?', axis=1)
    x = x[~conditionUndifined]
    
    return (normalize_data(x), normalize_data(y))


def get_stroke_data():
    # TODO: reformat text to number
    data = np.loadtxt("stroke.csv", delimiter=",", dtype=str)
    x = data[1:, 1:-1]
    y = data[1:, -1]
    return (normalize_data(x), normalize_data(y))


def normalize_data(x):
    return (x - np.mean(x, axis=0)) / np.var(x, axis=0)


def get_trash_column(x):
    return np.where(np.var(x, axis=0) == 0)[0]


if __name__ == "__main__":
    get_hypothyroid_data()
