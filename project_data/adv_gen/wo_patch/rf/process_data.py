import numpy as np
from numpy import genfromtxt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.linalg import norm
import csv
import cPickle

n_est = [1, 2, 5, 10, 20, 50]

epsilons = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

n_images = 400

# image_vecs = {}
# noise_vecs = {}
# correct_labels = {}
# accuracies = {}

accuracies = []
fitnesses = []
std_fitnesses = []
l2_norms = []
std_l2_norms = []

'''
Plots to be generated:
Accuracy (on adv. and on non-adv) vs. epsilon

'''

clfs = None

print("Opening classifiers")
with open("raw_data/clfs.pkl") as f:
    clfs = cPickle.load(f)
print("Opened classifiers")

for est in n_est:
    print("Estimator: {}".format(est))
    acc_vec = []
    fitness_vec = []
    std_fitness_vec = []
    l2_vec = []
    std_l2_vec = []

    for eps in epsilons:
        print("Epsilon: {}".format(eps))
        clf = clfs[str(est)]
        noises = genfromtxt("raw_data/noise/est_{}eps_{}.csv".format(est, eps), delimiter = ',')[:n_images]
        images = genfromtxt("raw_data/images/est_{}eps_{}.csv".format(est, eps), delimiter = ',')[:n_images]
        adv_images = images + noises
        true_labels = genfromtxt("raw_data/correct_labels/est_{}eps_{}.csv".format(est, eps), delimiter = ',')[1:n_images + 1] 
        non_adv_pred = clf.predict(images)
        adv_pred = clf.predict(adv_images)
        non_adv_probs = clf.predict_proba(images)
        adv_probs = clf.predict_proba(adv_images)
        no_images = images.shape[0]

        accuracy = accuracy_score(adv_pred[:n_images], true_labels[:n_images])

        prob_perturbation = [norm(prob) for prob in (adv_probs - non_adv_probs)]
        l2 = [norm(noise) for noise in noises]

        mean_fitness = np.mean(prob_perturbation)
        std_fitness = np.std(prob_perturbation)
        mean_l2 = np.mean(np.array(l2)
        std_l2 = np.std(np.array(l2)

    acc_vec.append(accuracy)
    fitness_vec.append(mean_fitness)
    std_fitness_vec.append(std_fitness)
    l2_vec.append(mean_l2)
    std_l2_vec.append(std_l2)

    accuracies.append(acc_vec)
    fitnesses.append(fitness_vec)
    std_fitnesses.append(std_fitness_vec)
    # l2_norms.append(mean_l2)
    # std_l2_norms.append(std_l2)

file = 'processed_data/accuracy.csv'
with open(file, 'w') as output:
    writer = csv.writer(output, delimiter = ',')
    writer.writerows(accuracies)
file = 'processed_data/fitness.csv'
with open(file, 'w') as output:
    writer = csv.writer(output, delimiter = ',')
    writer.writerows(fitnesses)
file = 'processed_data/std_fitness.csv'
with open(file, 'w') as output:
    writer = csv.writer(output, delimiter = ',')
    writer.writerows(std_fitnesses)
file = 'processed_data/l2.csv'
with open(file, 'w') as output:
    writer = csv.writer(output, delimiter = ',')
	writer.writerows(l2_norms)
file = 'processed_data/std_l2.csv'
with open(file, 'w') as output:
    writer = csv.writer(output, delimiter = ',')
	writer.writerows(std_l2_norms)
        
        


