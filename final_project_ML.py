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
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from Adaboost import Adaboost

# pip install matplotlib
# pip install tabulat
# pip install -U scikit-learn


############### ONE TIME FUNCTIONS START ###############


# This function sorts diabetes only once, since there are about 210 thousand lines
# of healthy people without diabetes (class 0), and only about 20k of class 2 and 3k of class 1
# This function should only be executed once, it outputs a new file sorted by the classes, and another file with indexes of where each class starts
# We do this to save in runtime, you don't need to run this function as we've provided you with the files.
def sort_diabetes_once():
    data = np.loadtxt("Datasets/diabetes.csv", delimiter=",", dtype=str)

    sorted_indices = np.argsort(data[:, 0])[::-1]

    # Sort the matrix based on the sorted indices
    sorted_matrix = data[sorted_indices]

    # Specify the output CSV file path
    output_file = "Datasets/diabetes_sorted.csv"
    out_index = "Datasets/diabetes_sorted_index.txt"

    # Write the sorted matrix to a CSV file
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(sorted_matrix)
    ind = [
        np.count_nonzero(sorted_matrix[:, 0] == "2.0"),
        np.count_nonzero(sorted_matrix[:, 0] == "2.0")
        + np.count_nonzero(sorted_matrix[:, 0] == "1.0"),
    ]
    print(ind)
    with open(out_index, "w") as file:
        file.write(str(ind[0]) + "\n" + str(ind))
    return ind


# Run this function only once after extracting the 50MB covid data file
# This function will output one clean file with 10k random lines from the big file with 1 million lines
def covid_random_5k():
    input_file = "Datasets/Covid Data.csv"
    output_file = "Datasets/covid.csv"

    num_lines_to_pick = 5000

    data = np.genfromtxt(input_file, delimiter=",", skip_header=1)
    data = np.delete(data, 4, axis=1)
    data = data[:, 0:-1]

    total_lines = data.shape[0]

    random_indices = np.random.choice(total_lines, num_lines_to_pick, replace=False)

    random_lines = []

    for idx in random_indices:
        random_lines.append(data[idx])

    random_lines = np.array(random_lines)

    np.savetxt(output_file, random_lines, delimiter=",", fmt="%s")


############### ONE TIME FUNCTIONS END ###############


def get_diabetes_data():
    data = np.loadtxt("Datasets/diabetes_sorted.csv", delimiter=",", dtype=str)
    num_rows = np.shape(data)[0]
    indexs = []
    # Get 3000 of random from class 0, 1000 of random class 1, and 1000 of random class 2.
    # This is fine because in reality there are more healthy people than people in class 1 or 2
    with open("Datasets/diabetes_sorted_index.txt", "r") as file:
        indexs = [line.strip() for line in file.readlines()]
    for i in range(len(indexs)):
        indexs[i] = int(indexs[i])
    class2, class1, class0 = (
        data[np.array(random.sample([*range(1, indexs[0])], 4500))],
        data[np.array(random.sample([*range(indexs[0] + 1, indexs[1])], 4500))],
        data[np.array(random.sample([*range(indexs[1], num_rows)], 5000))],
    )
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


def get_covid_data():
    data = np.loadtxt("Datasets/covid.csv", delimiter=",", dtype=float)
    x = data[:, :-1]
    y = data[:, -1]

    # 4 and above means no covid, 1-3 means covid to some degree
    # transform 4-7 to -1 and 1-3 to 1
    condition_positive = y >= 4.0
    y[condition_positive] = -1  # negative = no covid
    y[~condition_positive] = 1  # positive = yes covid

    return (normalize_data(x), y)


def get_heart_data():
    # get the data
    data = np.loadtxt("Datasets/heart.csv", delimiter=",", dtype=str)
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


def get_cancer_data():
    data = np.loadtxt("Datasets/lung_cancer.csv", delimiter=",", dtype=str)
    x = data[1:, 2:-1]
    y = data[1:, -1]

    # turn y into 1s and -1s
    # 1 means high likelyhood of lung cancer, -1 means low likelyhood of lung cancer
    likelyhoodMapping = {"High": 1, "Medium": 1, "Low": -1}
    v_mapping = np.vectorize(lambda y: likelyhoodMapping[y])
    y = v_mapping(y)

    # remove unnecessary data
    x = x.astype(float)
    y = y.astype(float)
    x = np.delete(x, get_trash_column(x), 1)

    return (normalize_data(x), y)


def get_stroke_data():
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
    smokeMapping = {"formerly smoked": 0, "Unknown": 1, "smokes": 2, "never smoked": 3}
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


################### TRAIN FUNCTIONS START ###################


# One vs One trainings
def train_ovo(X_train, X_test, y_train, y_test, c, max_iter, degree=1):
    # Add more dimensions to the model
    if degree > 1:
        X_train = get_poly(degree).fit_transform(X_train)
        X_test = get_poly(degree).fit_transform(X_test)

    logreg = LogisticRegression(max_iter=max_iter, C=c)

    # Create the one-vs-one classifier
    ovo = OneVsOneClassifier(logreg)

    # Train the model
    ovo.fit(X_train, y_train)

    # get precision, accuracy etc
    report = classification_report(y_test, ovo.predict(X_test), output_dict=True)

    return ovo, report, logreg


def train_GMM(x_train, x_test, y_train, y_test, k):
    # Create a GMM object
    # K is the number of components/clusters
    gmm = GaussianMixture(
        n_components=k,
    )

    # Fit the GMM model to your data
    gmm.fit(x_train, y_train)

    report = classification_report(y_test, gmm.predict(x_test), zero_division=1)

    return gmm, report


def train_adaboost(x_train, x_test, y_train, y_test, num_iter):
    models = [train_GMM(x_train, x_test, y_train, y_test, 2)[0], train_knn(x_train, x_test, y_train, y_test, 2)[0]]
    Cs, degrees = [10**-3, 10**-2, 10**-1, 1, 10, 10**2], [1, 2, 3]
    for c in Cs:
        for degree in degrees:
            models.append(train_ovo(x_train, x_test, y_train, y_test, c, 100, degree)[0])
    covidAdaModel = Adaboost(x_train, y_train)
    covidAdaModel.fit(num_iter, models=models)
    report = classification_report(y_test, covidAdaModel.predict(x_test))
    return covidAdaModel, report


def train_knn(x_train, x_test, y_train, y_test, k):
    # Create a k-NN object
    # k is the number of neighbors to consider
    knn = KNeighborsClassifier(n_neighbors=k)

    # Train the KNN model
    knn.fit(x_train, y_train)

    report = classification_report(
        y_test, knn.predict(x_test), zero_division=1, output_dict=True
    )

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


def main():
    # comment out the data you don't want to analyze
    analyze_covid_data()

    # analyze_cancer_data()

    # analyze_heart_data()

    # analyze_diabetes_data()




def analyze_covid_data():
    print("--------------- Analyzing Covid Data ---------------")

    x,y = get_covid_data()

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=45
    )

    # turn -1 to 0. it means the same thing, we're doing this
    # because otherwise GMM thinks there are 3 classes and it messes up the report
    # the -1s are for Adaboost
    y_train[y_train == -1.0] = 0
    y_test[y_test == -1.0] = 0

    ######## One vs One start ########
    print("++++++++++ One Vs One ++++++++++")
    print(
        "You should see Precision and Recall for the different C values plotted in a graph"
    )

    # find the best C for one vs all (the c is the complication constant)
    Cs = [0.01, 0.1, 1, 2, 5, 10]
    recall_0, recall_1, precision_0, precision_1 = [], [], [], []
    for c in Cs:
        model, report, logreg = train_ovo(x_train, x_test, y_train, y_test, c, 2000, 1)

        precision_0.append(float(report["0.0"]["precision"]))
        recall_0.append(float(report["0.0"]["recall"]))
        precision_1.append(float(report["1.0"]["precision"]))
        recall_1.append(float(report["1.0"]["recall"]))

    # Plot recall and precision for each class
    plt.xlabel("C")
    plt.ylabel("Precision / Recall")
    plt.title(
        "One Vs One\nPrecision and Recall for different C values for both classes"
    )

    plt.plot(Cs, precision_0, label="Precision (class 0)")
    plt.plot(Cs, recall_0, label="Recall (class 0)")
    plt.plot(Cs, precision_1, label="Precision (class 1)")
    plt.plot(Cs, recall_1, label="Recall (class 1)")

    plt.legend()
    plt.show()

    print("The optimal C value seems to be C = 1\n")
    ######## One vs One using Linear Regression end ########

    ######## GMM start ########        
    print("\n++++++++++ GMM ++++++++++")
    print("class 0 = doesn't have covid\nclass 1 = has covid")
    print("Let's try different numbers of classes and fit the GMM for each")

    cost_values = []
    for k in range(1, 11):
        gmm, report = train_GMM(x_train, x_test, y_train, y_test, k)
        cost_values.append(gmm.score(x_train))

    plt.plot(range(1, 11), cost_values, marker="o")
    plt.xlabel("Number of classes")
    plt.ylabel("Cost function")
    plt.title("GMM\nCost function of Gaussian Mixture Model")
    plt.grid(True)
    plt.show()

    print("Highest Jumps seem to be at K = 4 and K = 8. and they are all spaces pretty equally.")
    print("Precision and Recall for K = 2:")

    gmm, report = train_GMM(x_train, x_test, y_train, y_test, 2)
    print(report)
    ######## GMM end ########

    ######## KNN start ########
    print("++++++++++ KNN ++++++++++")
    print("class 0 = doesn't have covid\nclass 1 = has covid")
    print(
        "Let's try different number of neighbors to find the optimal K based on the F1 Score"
    )
    Ks = [1, 2, 5, 10, 20, 50, 100, 200]

    # go back go -1, 1 since KNN doesn't get confused like GMM
    f1_scores_0 = []
    f1_scores_1 = []
    f1_scores_sum = []
    for k in Ks:
        knn, report = train_knn(x_train, x_test, y_train, y_test, k)
        f1_scores_0.append(float(report["0.0"]["f1-score"]))
        f1_scores_1.append(float(report["1.0"]["f1-score"]))
        f1_scores_sum.append(f1_scores_0[-1] + f1_scores_1[-1])

    # Plot recall and precision for each class
    plt.xlabel("K")
    plt.ylabel("F1 Score")
    plt.title("KNN\nF1 Score for different K values for both classes")

    plt.plot(Ks, f1_scores_0, label="F1 Score (class 0)")
    plt.plot(Ks, f1_scores_1, label="F1 Score (class 1)")
    plt.plot(Ks, f1_scores_sum, label="Sum (classes 1,0)")

    plt.legend()
    plt.show()

    print("As we can see 1 <= K <= 10 gives the best results.\nPrecision and Recall for K = 1:")
    knn, report = train_knn(x_train, x_test, y_train, y_test, 1)
    print(report)
    ######## KNN end ########

    ######## Adaboost start ########

    print("++++++++++ Adaboost ++++++++++")
    adaboost,report = train_adaboost(x_train, x_test, y_train, y_test, 10)
    print(report)

    ######## Adaboost end ########


def analyze_cancer_data():
    print("--------------- Analyzing Covid Data ---------------")

    x,y = get_cancer_data()

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=45
    )

    # turn -1 to 0. it means the same thing, we're doing this
    # because otherwise GMM thinks there are 3 classes and it messes up the report
    # the -1s are for Adaboost
    y_train[y_train == -1.0] = 0
    y_test[y_test == -1.0] = 0

    ######## One vs One start ########
    print("++++++++++ One Vs One ++++++++++")
    print(
        "You should see Precision and Recall for the different C values plotted in a graph"
    )

    # find the best C for one vs all (the c is the complication constant)
    Cs = [0.01, 0.1, 1, 2, 5, 10]
    recall_0, recall_1, precision_0, precision_1 = [], [], [], []
    for c in Cs:
        model, report, logreg = train_ovo(x_train, x_test, y_train, y_test, c, 2000, 1)

        precision_0.append(float(report["0.0"]["precision"]))
        recall_0.append(float(report["0.0"]["recall"]))
        precision_1.append(float(report["1.0"]["precision"]))
        recall_1.append(float(report["1.0"]["recall"]))

    # Plot recall and precision for each class
    plt.xlabel("C")
    plt.ylabel("Precision / Recall")
    plt.title(
        "One Vs One\nPrecision and Recall for different C values for both classes"
    )

    plt.plot(Cs, precision_0, label="Precision (class 0)")
    plt.plot(Cs, recall_0, label="Recall (class 0)")
    plt.plot(Cs, precision_1, label="Precision (class 1)")
    plt.plot(Cs, recall_1, label="Recall (class 1)")

    plt.legend()
    plt.show()

    print("The optimal C value seems to be C >= 2\n")
    ######## One vs One using Linear Regression end ########

    ######## GMM start ########        
    print("\n++++++++++ GMM ++++++++++")
    print("class 0 = doesn't have covid\nclass 1 = has covid")
    print("Let's try different numbers of classes and fit the GMM for each")

    cost_values = []
    for k in range(1, 11):
        gmm, report = train_GMM(x_train, x_test, y_train, y_test, k)
        cost_values.append(gmm.score(x_train))

    plt.plot(range(1, 11), cost_values, marker="o")
    plt.xlabel("Number of classes")
    plt.ylabel("Cost function")
    plt.title("GMM\nCost function of Gaussian Mixture Model")
    plt.grid(True)
    plt.show()

    print("\nAs we can see the best K is K = 2 (as expected)")
    print("Precision and Recall for K = 2:")

    gmm, report = train_GMM(x_train, x_test, y_train, y_test, 2)
    print(report)
    ######## GMM end ########

    ######## KNN start ########
    print("++++++++++ KNN ++++++++++")
    print("class 0 = doesn't have covid\nclass 1 = has covid")
    print(
        "Let's try different number of neighbors to find the optimal K based on the F1 Score"
    )
    Ks = [1, 2, 5, 10, 20, 50, 100, 200]

    # go back go -1, 1 since KNN doesn't get confused like GMM
    f1_scores_0 = []
    f1_scores_1 = []
    f1_scores_sum = []
    for k in Ks:
        knn, report = train_knn(x_train, x_test, y_train, y_test, k)
        f1_scores_0.append(float(report["0.0"]["f1-score"]))
        f1_scores_1.append(float(report["1.0"]["f1-score"]))
        f1_scores_sum.append(f1_scores_0[-1] + f1_scores_1[-1])

    # Plot recall and precision for each class
    plt.xlabel("K")
    plt.ylabel("F1 Score")
    plt.title("KNN\nF1 Score for different K values for both classes")

    plt.plot(Ks, f1_scores_0, label="F1 Score (class 0)")
    plt.plot(Ks, f1_scores_1, label="F1 Score (class 1)")
    plt.plot(Ks, f1_scores_sum, label="Sum (classes 1,0)")

    plt.legend()
    plt.show()
    ######## KNN end ########

    ######## Adaboost start ########

    print("++++++++++ Adaboost ++++++++++")
    adaboost, report = train_adaboost(x_train, x_test, y_train, y_test, 10)

    print(report)

    ######## Adaboost end ########


def analyze_heart_data():
    return


def analyze_diabetes_data():
    return


if __name__ == "__main__":
    main()
