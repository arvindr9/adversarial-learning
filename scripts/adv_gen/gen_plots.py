import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt

n_est = [1, 2, 5, 10, 20, 50, 100]

epsilons = [0.0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

n_images = 500

accuracies = genfromtxt("processed_data/accuracy.csv", delimiter = ',')
fitnesses = genfromtxt("processed_data/fitness.csv", delimiter = ',')
std_fitnesses = genfromtxt("processed_data/std_fitness.csv", delimiter = ',')

colors = ['r-', 'b-', 'g-', 'y-', 'p-', 'k-', 'bo-', 'c-']

f1, ax1 = plt.subplots()
mean_fitness = []
for i in range(len(n_est)):
    data = [round(float(f), 3) for f in fitnesses[i]]
    mean_fitness.append(epsilons)
    mean_fitness.append(data)
    mean_fitness.append(colors[i])
ax1.plot(*mean_fitness)
plt.legend([f"Estimators: {est}" for est in n_est])
plt.title('Mean Optimal fitness vs epsilon')
plt.xlabel('Epsilon')
plt.ylabel('Probability perturbation (L2 norm)')
plt_name=  'plots/plot_fitness.png'
plt.savefig(plt_name)

f2, ax2 = plt.subplots()
accuracy_vec = []
for i in range(len(n_est)):
    data = [round(float(f), 3) for f in accuracies[i]]
    accuracy_vec.append(epsilons)
    accuracy_vec.append(data)
    accuracy_vec.append(colors[i])
ax2.plot(*accuracy_vec)
plt.legend([f"Estimators: {est}" for est in n_est])
plt.title('Accuracy vs epsilon')
plt.xlabel('Epsilon')
plt.ylabel('Accuracy')
plt_name=  'plots/accuracy.png'
plt.savefig(plt_name)

