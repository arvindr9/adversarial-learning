import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import csv
import pickle

n_est = [1, 2, 5, 10, 20, 50, 100]

epsilons = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1]

image_vecs = {}
noise_vecs = {}
correct_labels = {}
accuracies = {}

clfs = None

with open("raw_data/clfs.pkl") as f:
    clfs = pickle.load(f)


for est in n_est:
    for eps in epsilons:
        clf = clfs[str(est)]
        noises = genfromtxt("raw_data/noise/est_{}eps_{}".format(est, eps))
        images = genfromtxt("raw_data/images/est_{}eps_{}".format(est, eps))
        true_labels = genfromtxt("raw_data/correct_labels/est_{}eps_{}".format(est, eps))
        noise_vecs["{}_{}".format(est, eps)] = noises
        image_vecs["{}_{}".format(est, eps)] = images
        correct_labels["{}_{}".format(est, eps)] = true_labels
        non_adv_pred = clf.predict(images)
        adv_pred = clf.predict(images + noises)
        no_images = images.shape[0]
        accuracy = clf.accuracy_score(correct_labels, adv_pred)
        accuracies["{}_{}".format(est, eps)] = accuracy
        
        


