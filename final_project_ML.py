from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tabulate import tabulate
import numpy as np
import csv
import random
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier


# This function sorts diabetes only once, since there are about 210 thousand lines
# of healthy people without diabetes (class 0), and only about 20k of class 2 and 3k of class 1
# This function should only be executed once, it outputs a new file sorted by the classes, and another file with indexes of where each class starts
# We do this to save in runtime, you don't need to run this function as we've provided you with the files.
def sort_diabetes_once():
    data = np.loadtxt("diabetes.csv", delimiter=",", dtype=str)

    sorted_indices = np.argsort(data[:, 0])[::-1]

    # Sort the matrix based on the sorted indices
    sorted_matrix = data[sorted_indices]

    # Specify the output CSV file path
    output_file = "diabetes_sorted.csv"
    out_index = "diabetes_sorted_index.txt"

    # Write the sorted matrix to a CSV file
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(sorted_matrix)
    ind = [np.count_nonzero(sorted_matrix[:, 0] == '2.0'), np.count_nonzero(
        sorted_matrix[:, 0] == '2.0')+np.count_nonzero(sorted_matrix[:, 0] == '1.0')]
    print(ind)
    with open(out_index, 'w') as file:
        file.write(str(ind[0]) + '\n'+str(ind))
    return ind


def get_diabetes_data():
    data = np.loadtxt("diabetes_sorted.csv", delimiter=",", dtype=str)
    num_rows = np.shape(data)[0]
    indexs = []
    # Get 3000 of random from class 0, 1000 of random class 1, and 1000 of random class 2.
    # This is fine because in reality there are more healthy people than people in class 1 or 2
    with open('diabetes_sorted_index.txt', 'r') as file:
        indexs = [line.strip() for line in file.readlines()]
    for i in range(len(indexs)):
        indexs[i] = int(indexs[i])
    class2, class1, class0 = data[np.array(random.sample([*range(1, indexs[0])], 4500))], data[np.array(random.sample(
        [*range(indexs[0]+1, indexs[1])], 4500))], data[np.array(random.sample([*range(indexs[1], num_rows)], 4500))]
    data = np.concatenate((data[0:1,], class2, class1, class0), axis=0)

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
    y = y[~conditionUndifined]

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
    y = y[~conditionUndifined]

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
    return np.where(np.var(x, axis=0) <= 0.001)[0]


def draw_cost_function(model, x_train, y_train):
    # Retrieve the decision function values
    decision_values = np.asarray(model.decision_function(x_train))

    # Calculate the softmax of the decision function values
    softmax = np.exp(decision_values) / \
        np.sum(np.exp(decision_values), axis=0, keepdims=True)

    # Calculate the loss function values (negative log likelihood)
    loss_values = -np.log(softmax[np.arange(len(y_train)), y_train])

    # Plot the loss function values
    plt.plot(loss_values)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss Function Progression')
    plt.show()


################### TRAIN FUNCTIONS START ###################
# One vs One training
def train_ovo(X_train, X_test, y_train, y_test, c, max_iter, degree=1):

    # Add more dimensions to the model
    if (degree > 1):
        X_train = get_poly(degree).fit_transform(X_train)
        X_test = get_poly(degree).fit_transform(X_test)

    logreg = LogisticRegression(max_iter=max_iter, C=c)

    # Create the one-vs-one classifier
    ovo = OneVsOneClassifier(logreg)

    # Train the model
    ovo.fit(X_train, y_train)

    # get precision, accuracy etc
    report = classification_report(
        y_test, ovo.predict(X_test), output_dict=True)

    return ovo, report, logreg


# GMM training
def train_GMM(x_train, x_test, y_train, y_test, k):

    # Create a GMM object
    # K is the number of components/clusters
    gmm = GaussianMixture(n_components=k)

    # Fit the GMM model to your data
    gmm.fit(x_train, y_train)

    report = classification_report(y_test, gmm.predict(x_test))

    return gmm, report


def train_knn(x_train, x_test, y_train, y_test, k):
    # Create a k-NN object
    # k is the number of neighbors to consider
    knn = KNeighborsClassifier(n_neighbors=k)

    # Train the KNN model
    knn.fit(x_train, y_train)

    report = classification_report(y_test, knn.predict(x_test))

    return knn, report
################### TRAIN FUNCTIONS END ###################


def compare_arrays(arr1, arr2):
    headers = ["Index", "Array 1", "Array 2", "Comparison"]
    table_data = []

    for i in range(len(arr1)):
        comparison = "Equal" if arr1[i] == arr2[i] else "Not equal"
        table_data.append([i, arr1[i], arr2[i], comparison])

    table = tabulate(table_data, headers, tablefmt="grid")
    print(table)


def get_poly(n):
    return PolynomialFeatures(degree=n, interaction_only=False, include_bias=False)


if __name__ == "__main__":

    ################# heart start #################
    # Get the data
    x, y = get_heart_data()

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=45)

    ######## One vs One using Linear Regression start ########
    print("++++++++++ One Vs One ++++++++++")

    # find the best C
    Cs = [0.01, 0.1, 1, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    recall_0, recall_1, precision_0, precision_1 = [], [], [], []
    for c in Cs:
        print("C = " + str(c) + ":")
        model, report, logreg = train_ovo(
            x_train, x_test, y_train, y_test, c, 10000, 1)

        precision_0.append(float(report['0.0']['precision']))
        recall_0.append(float(report['0.0']['recall']))
        precision_1.append(float(report['1.0']['precision']))
        recall_1.append(float(report['1.0']['recall']))

    plt.plot(Cs, precision_0)
    plt.plot(Cs, recall_0)
    plt.show()
    plt.plot(Cs, precision_1)
    plt.plot(Cs, recall_1)
    plt.show()

    # Draw cost function of the linear regression model for the best C

    model, report, logreg = train_ovo(
        x_train, x_test, y_train, y_test, c, 10000, 1)

    draw_cost_function(model, x_train, y_train)
    ######## One vs One using Linear Regression end ########

    ######## GMM start ########
    print("++++++++++ GMM ++++++++++")
    gmm, report = train_GMM(x_train, x_test, y_train, y_test, 2)
    print(report)
    ######## GMM end ########

    ######## KNN start ########
    print("++++++++++ KNN ++++++++++")
    knn, report = train_knn(x_train, x_test, y_train, y_test, 500)
    print(report)
    ######## KNN end ########

    ################# heart end #################

    ################# diabetes start #################

    ################# diabetes end #################

# sort_diabetes_once()
