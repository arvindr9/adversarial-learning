from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
from keras.datasets import cifar10
import time
import argparse

no_threads = 20

parser = argparse.ArgumentParser()

parser.add_argument("--threads", default = no_threads, help = "number of threads", type = int)

args = parser.parse_args()
arguments = args.__dict__

#Make a dictionary of all arguments
args_dict = {k: v for k,v in arguments.items()}

no_threads = args_dict['threads']


n_estimators = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
max_depths = [5, 10, 20, 50, 100, 200, 500, 1000]

accuracies = {est: {} for est in n_estimators}
times = {est: {} for est in n_estimators}



def recordData(est, depth):
    print(est, depth)
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2] * X_train.shape[3])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2] * X_test.shape[3])
    clf = RandomForestClassifier(n_estimators=est, max_depth=depth)
    t_start = time.time()
    clf.fit(X_train, y_train)
    t_end = time.time()
    accuracy = accuracy_score(clf.predict(X_test), y_test)
    accuracies[est][depth] = accuracy
    total_time = t_end - t_start
    times[est][depth] = total_time
    print("time:", time, "accuracy:", accuracy)

tasks = []
for est in n_estimators:
    for depth in max_depths:
        recordData(est, depth)
        # tasks.append((est, depth))
# Parallel(n_jobs = no_threads)(delayed(recordData)((est, depth) for (est, depth) in tasks))

print("times:", times)
print("accuracies:", accuracies)
