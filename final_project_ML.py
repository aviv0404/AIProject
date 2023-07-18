from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from tabulate import tabulate
import numpy as np
import csv


# This function sorts diabetes only once, since there are 
def sort_diabetes_once():
    data = np.loadtxt("diabetes.csv", delimiter=",", dtype=str)
    
    sorted_indices = np.argsort(data[:, 0])[::-1]

    # Sort the matrix based on the sorted indices
    sorted_matrix = data[sorted_indices]

    # Specify the output CSV file path
    output_file = "diabetes_sorted.csv"

    # Write the sorted matrix to a CSV file
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(sorted_matrix)
    

def get_diabetes_data():
    # get only a portion of the data, remove top row of column names
    data = np.loadtxt("diabetes.csv", delimiter=",", dtype=str)
    num_rows = np.shape(data)[0]
    random_indices = np.random.choice(num_rows, size=500, replace=False)
    data = data[random_indices]

    x = data[1:, 1:]
    x = x.astype(float)

    # deleting unecessary data with variance = 0
    x = np.delete(x, get_trash_column(x), 1)

    # y is first column
    y = data[1:, 0]
    y = y.astype(float)

    # return normalized data
    return (normalize_data(x), y)


def get_heart_data():
    # get the data
    data = np.loadtxt("heart.csv", delimiter=",", dtype=str)
    x = data[1:, :-1]
    y = data[1:, -1]

    # reformat text to numbers

    # let's see which columns need reformatting:
    # print_unique_values_per_col(x) # uncomment to see

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

    return (normalize_data(x), y)


def print_unique_values_per_col(x):
    column_sets = [set(column) for column in np.transpose(x)]
    for column in range(len(column_sets)):
        print(str(column) + ": " + str(column_sets[column]) + "\n")


def get_hypothyroid_data():
    data = np.loadtxt("hypothyroid.csv", delimiter=",", dtype=str)
    x = data[1:, :-1]
    y = data[1:, -1]
    # remove unrelevent column
    x = np.delete(x, 27, 1)
    # Remove the rows that contain '?'
    conditionUndifined = np.any(x == "?", axis=1)
    x = x[~conditionUndifined]

    # let's see which columns need reformatting:
    # print_unique_values_per_col(x) # uncomment to see

    # we need to work on columns: 1 to 16, 18, 20, 22, 24, 26, 27

    # column 1: Sex
    condition_female = x[:, 1] == "F"
    # Change 'F' to 0 and 'M' to 1 in column 1
    x[condition_female, 1] = 0
    x[~condition_female, 1] = 1

    # change all t to 1 and f to 0
    condition_f = x == "f"
    condition_t = x == "t"
    x[condition_f] = 0
    x[condition_t] = 1

    # last column
    ecgMapping = {"SVHD": 0, "SVHC": 1, "STMW": 2, "SVI": 3, "other": 4}
    v_mapping = np.vectorize(lambda x: ecgMapping[x])
    x[:, -1] = v_mapping(x[:, -1])

    # fix y column
    # # change all t to 1 and f to 0
    condition_P = y == "P"
    y[condition_P] = 1
    y[~condition_P] = 0

    # let's remove unecessary data (variance = 0)
    x = x.astype(float)
    y = y.astype(float)
    x = np.delete(x, get_trash_column(x), 1)

    # return normalized data
    return (normalize_data(x), y)


def get_stroke_data():
    # TODO: reformat text to number
    data = np.loadtxt("stroke.csv", delimiter=",", dtype=str)
    x = data[1:, 1:-1]
    y = data[1:, -1]

    # let's see which columns need to be worked on
    # print_unique_values_per_col(x)
    conditionUndifined = np.any(x == "N/A", axis=1)
    x = x[~conditionUndifined]
    # we need to work on columns: 0, 4, 5, 6, 9

    # column 0
    sexMapping = {"Male": 0, "Other": 1, "Female": 2}
    v_mapping = np.vectorize(lambda x: sexMapping[x])
    x[:, 0] = v_mapping(x[:, 0])

    # column 4
    condition_yes = x[:, 4] == "Yes"
    x[condition_yes, 4] = 1
    x[~condition_yes, 4] = 0

    # column 5
    workMapping = {
        "children": 0,
        "Never_worked": 1,
        "Govt_job": 2,
        "Self-employed": 3,
        "Private": 4,
    }

    v_mapping = np.vectorize(lambda x: workMapping[x])
    x[:, 5] = v_mapping(x[:, 5])

    # column 6
    condition_yes = x[:, 6] == "Urban"
    x[condition_yes, 6] = 1
    x[~condition_yes, 6] = 0

    # column 9: smoking status
    smokeMapping = {"formerly smoked": 0,
                    "Unknown": 1, "smokes": 2, "never smoked": 3}
    v_mapping = np.vectorize(lambda x: smokeMapping[x])
    x[:, 9] = v_mapping(x[:, 9])

    x = x.astype(float)
    y = y.astype(float)

    x = np.delete(x, get_trash_column(x), 1)

    # remove id column
    x = np.delete(x, 0, axis=1)

    return (normalize_data(x), y)


def normalize_data(x):
    return (x - np.mean(x, axis=0)) / np.var(x, axis=0)


def get_trash_column(x):
    return np.where(np.var(x, axis=0) == 0)[0]


def train_ovo(x, y, c, cv):
    logreg = LogisticRegression()
    # Create the grid search object
    grid_search = GridSearchCV(logreg, c, cv=cv)

    # Fit the grid search to the training data
    grid_search.fit(x, y)

    # Make predictions on the test set using the best model
    all_models = grid_search.cv_results_
    best_model = grid_search.best_estimator_
    # Print the classification report
    report = classification_report(y, best_model.predict(x))
    return best_model, report, grid_search


def compare_arrays(arr1, arr2):
    headers = ["Index", "Array 1", "Array 2", "Comparison"]
    table_data = []

    for i in range(len(arr1)):
        comparison = "Equal" if arr1[i] == arr2[i] else "Not equal"
        table_data.append([i, arr1[i], arr2[i], comparison])

    table = tabulate(table_data, headers, tablefmt="grid")
    print(table)


if __name__ == "__main__":

    # ################# heart data #################
    # x, y = get_diabetes_data()

    # # Define the parameter grid for grid search
    # c = {'C': [0.001, 0.01, 0.05, 0.1, 1.0, 10.0]}

    # model, report, grid_search = train_ovo(x, y, c, 5)

    # predicted = model.predict(x)

    # # compare_arrays(y, predicted)

    # # Print the best parameter and the corresponding score
    # print("Best C parameter:", grid_search.best_params_['C'])
    # print("Best score:", grid_search.best_score_)

    # print(report)

    ################# heart data end #################

    sort_diabetes_once()
