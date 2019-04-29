from numpy import genfromtxt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class AdversarialTest():

    def __init__(self, config):
        self.epsilons = config['epsilons']
        self.base_classifier = config['base_classifier']
        self.processed_data_path = config['processed_data_path']
        self.plot_path = config['plot_path']
        if self.base_classifier == 'random_forest':
            self.param_name = 'Estimators'
            self.param_vals = config['n_estimators']
        if self.base_classifier == 'svm':
            self.param_name = 'C'
            self.param_vals = config['C_vals']
        self.file_type = config['file_type']
        
        self.main()

    def main(self):
        n_params = len(self.param_vals)
        accuracies = genfromtxt("{}/accuracy.csv".format(self.processed_data_path), delimiter = ',')
        fitnesses = genfromtxt("{}/fitness.csv".format(self.processed_data_path), delimiter = ',')
        std_fitnesses = genfromtxt("{}/std_fitness.csv".format(self.processed_data_path), delimiter = ',')

        colors = ['r-', 'b-', 'g-', 'y-', 'p-', 'k-', 'bo-', 'c-', 'm-'][:n_params]

        f1, ax1 = plt.subplots()
        mean_fitness = []
        for i in range(len(self.param_vals)):
            data = [round(float(f), 3) for f in fitnesses[i]]
            mean_fitness.append(self.epsilons)
            mean_fitness.append(data)
            mean_fitness.append(colors[i])
        ax1.plot(*mean_fitness)
        plt.legend(["{}: {}".format(self.param_name, param) for param in self.param_vals])
        plt.title('Mean Optimal fitness vs epsilon')
        plt.xlabel('Epsilon')
        plt.ylabel('Probability perturbation (L2 norm)')
        plt_name=  '{}/plot_fitness.{}'.format(self.plot_path, self.file_type)
        plt.savefig(plt_name)

        f2, ax2 = plt.subplots()
        accuracy_vec = []
        for i in range(len(self.param_vals)):
            data = [round(float(f), 3) for f in accuracies[i]]
            accuracy_vec.append(self.epsilons)
            accuracy_vec.append(data)
            accuracy_vec.append(colors[i])
        ax2.plot(*accuracy_vec)
        plt.legend(["{}: {}".format(self.param_name, param) for param in self.param_vals])
        plt.title('Accuracy vs epsilon')
        plt.xlabel('Epsilon')
        plt.ylabel('Accuracy')
        plt_name=  '{}/accuracy.{}'.format(self.plot_path, self.file_type)
        plt.savefig(plt_name)