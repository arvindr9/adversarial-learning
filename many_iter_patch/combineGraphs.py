import matplotlib.pyplot as plt
import csv
import numpy as np

entries = {"mean_fitness": 0, "mean_noise": 1, "var_noise": 2,
        "min_noise": 3, "max_noise": 4, "wrong_output": 5,
        "mean_l0": 6, "min_l0": 7, "max_l0": 8, "var_l0": 9,
        "mean_l1": 10, "min_l1": 11, "max_l1": 12, "var_l1": 13}

data = {}

epsilons = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
estimators = [1, 2, 4, 5, 10]


i = -1
for estimator in estimators:
    i += 1
    data[estimator] = {}
    f = f"plots/data{estimator}.csv"
    with open(f, newline='\n') as csvfile:
        reader = list(csv.reader(csvfile, delimiter=','))
        for entry in entries:
            #print(f"sz: {len(reader[entries[entry]])}")
            print("len", len(reader), "i", entries[entry])
            data[estimator][entry] = [round(float(reader[i][entries[entry]]), 3) for i in range(len(estimators))]
for k, v in data.items():
    print(k)
    for k2, v2 in v.items():
        print(k2, v2)

colors = ['r-', 'b-', 'g-', 'y-', 'p-', 'k-', 'bo-', 'c-']

f1, ax1 = plt.subplots()
mean_fitness = []
i = 0
for est in estimators:
    mean_fitness.append(epsilons)
    mean_fitness.append(data[est]["mean_fitness"])
    mean_fitness.append(colors[i])
    i += 1
ax1.plot(*mean_fitness)
plt.legend([f"Estimators: {est}" for est in estimators])
plt.title('Mean Optimal fitness vs epsilon')
plt.xlabel('Epsilon')
plt.ylabel('Mean Optimal fitness values')
plt_name=  'plot_fitness.png'
plt.savefig(plt_name)

f2, ax2 = plt.subplots()
mean_noise = []
i = 0
for est in estimators:
    mean_noise.append(epsilons)
    mean_noise.append(data[est]["mean_noise"])
    mean_noise.append(colors[i])
    i += 1
ax2.plot(*mean_noise)
plt.legend([f"Estimators: {est}" for est in estimators])
plt.title('Mean Optimal Noise vs epsilon')
plt.xlabel('Epsilon')
plt.ylabel('Mean Optimal Noise (L2 norm)')
plt_name=  'plot_noise.png'
plt.savefig(plt_name)


f3, ax3 = plt.subplots()
mean_noise = []
i = 0
for est in estimators:
    m_noise = data[est]["mean_noise"]
    v_noise = data[est]["var_noise"]
    ax3.plot(epsilons, m_noise, colors[i])
    ax3.fill_between(epsilons, list(np.ndarray.flatten(np.array(m_noise) + 0.2 * np.array(v_noise))), list(np.ndarray.flatten(np.array(m_noise) - 0.2 * np.array(v_noise))))
    i += 1
plt.legend([f"Estimators: {est}" for est in estimators])
plt.title('Optimal noise distribution vs epsilon')
plt.xlabel('Epsilon')
plt.ylabel('Optimal noise distribution (L2 norm)')
plt_name = 'plot_noise_distribution.png'
plt.savefig(plt_name)

f4, ax4 = plt.subplots()
mean_l0 = []
i = 0
for est in estimators:
    mean_l0.append(epsilons)
    mean_l0.append(data[est]["mean_l0"])
    mean_l0.append(colors[i])
    i += 1
ax4.plot(*mean_l0)
plt.legend([f"Estimators: {est}" for est in estimators])
plt.title('Mean optimal L0 norm vs epsilon')
plt.xlabel('Epsilon')
plt.ylabel('Mean Optimal L0 norm')
plt_name=  'plot_l0.png'
plt.savefig(plt_name)

f5, ax5 = plt.subplots()
mean_l1 = []
i = 0
for est in estimators:
    mean_noise.append(epsilons)
    mean_noise.append(data[est]["mean_l1"])
    mean_noise.append(colors[i])
    i += 1
ax5.plot(*mean_noise)
plt.legend([f"Estimators: {est}" for est in estimators])
plt.title('Mean optimal L1 norm vs epsilon')
plt.xlabel('Epsilon')
plt.ylabel('Mean Optimal L1 norm')
plt_name=  'plot_l1.png'
plt.savefig(plt_name)

f6, ax6 = plt.subplots()
wrong_output = []
i = 0
for est in estimators:
    wrong_output.append(epsilons)
    wrong_output.append(data[est]["wrong_output"])
    wrong_output.append(colors[i])
    i += 1
ax6.plot(*wrong_output)
plt.legend([f"Estimators: {est}" for est in estimators])
plt.title('Percentage misclassified vs epsilon')
plt.xlabel('Epsilon')
plt.ylabel('Percentage misclassified')
plt_name=  'plot_misclassified.png'
plt.savefig(plt_name)




