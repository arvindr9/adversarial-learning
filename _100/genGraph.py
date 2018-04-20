import csv
import matplotlib.pyplot as plt

rows = list(csv.reader(open('accuracy.csv', 'r')))
r0 = list(map(float, rows[0]))
r1 = list(map(float, rows[1]))

epsilons = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
f1, ax1 = plt.subplots()
ax1.plot(epsilons, r0, 'b')
ax1.plot(epsilons, r1, 'r')
plt.legend(['Trained RF', 'Untrained RF'])
plt.xlabel('Epsilon')
plt.ylabel('Accuracy')
plt_name = 'iter_10_est_100.png'
plt.savefig(plt_name)