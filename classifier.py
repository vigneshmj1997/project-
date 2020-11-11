import argparse
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import plot_confusion_matrix


def preprocess(file):
    data = pd.read_csv(file)
    data["Decision"] = data["Decision"].replace(
        {"VERY_FRESH": 0, "EARLY_SPOILED": 1, "HALF_SPOILED": 2, "FULL_SPOILED": 3}
    )
    X = np.array(data[["S1", "S2", "S3", "S4", "S5", "S6"]])
    Y = np.array(data["Decision"])

    return X, Y


def train(clf, X_train, y_train, X_test, y_test, name, log):

    print("Traning the model")
    clf.fit(X_train, y_train)
    print("Trained model")
    score = clf.score(X_test, y_test)
    log.write(str(score) + "," + name + ",\n")
    print("Score ", score)
    plot_confusion_matrix(clf, X_test, y_test)
    if not os.path.isdir("images/confusion"):
        os.mkdir("images/confusion")
    plt.title(name)    
    plt.savefig(os.getcwd()+"/images/confusion/"+csvFileName+"/"+name.replace(" ","_")+".png")
    
def main():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--train", help="location of train csv file location")
    parser.add_argument("--test", help="location of  test csv file location")
    parser.add_argument("--log", help="log file", default="log.txt")
    args = parser.parse_args()
    if not os.path.isdir("images"):
        os.mkdir("images")
    X_train, y_train = preprocess(args.train)
    x_test, y_test = preprocess(args.test)
    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
    ]
    names = [
        "Nearest Neighbors",
        "Linear SVM",
        "RBF SVM",
        "Gaussian Process",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
        "Naive Bayes",
        "QDA",
    ]
    log = open(str(args.log), "w")
    for i in range(len(classifiers)):
        print("Classifier", names[i])
        try:
            train(classifiers[i], X_train, y_train, X_test, y_test, names[i], log)
        except Exception as e:
            print("Error in ", names[i], e)

    log.close()


if __name__ == "__main__":
    main()
