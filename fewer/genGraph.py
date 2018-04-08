import csv
import matplotlib.pyplot as plt

rows = list(csv.reader(open('accuracy.csv', 'r')))

f1, ax1 = plt.subplots()
epsilons = [i * 0.1 for i in range (1, 11)]
ax1.plot(epsilons, rows[0], 'b', epsilons, rows[1], 'o')
plt.legend(['Trained RF', 'Untrained RF'])
plt.xlabel('Epsilon')
plt.ylabel('Accuracy')
plt_name = 'iter_1_est_20.png'
plt.savefig(plt_name)
