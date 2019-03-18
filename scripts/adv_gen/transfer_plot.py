from numpy import genfromtxt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class TransferPlot():

    def __init__(self, config):
        self.epsilons = config['epsilons']
        self.base_classifier = config['base_classifier']
        self.transfer_data_path = config['transfer_data_path']
        self.transfer_plot_path = config['transfer_plot_path']
        if self.base_classifier == 'random_forest':
            self.file_head = 'n_estimators'
            self.param_name = "Estimators"
            self.param_vals = config['n_estimators']
        elif self.base_classifier == 'svm':
            self.file_head = 'C'
            self.param_name = 'C'
            self.param_vals = config['C_vals']
        self.file_type = config['file_type']
        self.main()

    def main(self):

        n_params = len(self.param_vals)
	print(self.param_vals)

        for param in self.param_vals:
            
        
            accuracies = genfromtxt('{}/{}_{}_accuracy.csv'.format(self.transfer_data_path, self.file_head, param), delimiter = ',')
            fitnesses = genfromtxt('{}/{}_{}_fitness.csv'.format(self.transfer_data_path, self.file_head, param), delimiter = ',')
            std_fitnesses = genfromtxt('{}/{}_{}_std_fitness.csv'.format(self.transfer_data_path, self.file_head, param), delimiter = ',')

            colors = ['r-', 'b-', 'g-', 'y-', 'p-', 'k-', 'bo-', 'c-', 'm-'][:n_params]

            f1, ax1 = plt.subplots()
            mean_fitness = []
            for i in range(len(self.param_vals)):
                data = [round(float(f), 3) for f in fitnesses[i]]
                mean_fitness.append(self.epsilons)
                mean_fitness.append(data)
                mean_fitness.append(colors[i])
            ax1.plot(*mean_fitness)
            plt.legend(["{}: {}".format(self.param_name, p) for p in self.param_vals])
            plt.title('Mean Optimal fitness vs epsilon ({} = {})'.format(self.param_name, param))
            plt.xlabel('Epsilon')
            plt.ylabel('Probability perturbation (L2 norm)')
            plt_name=  '{}/{}_{}_plot_fitness.{}'.format(self.transfer_plot_path, self.file_head, param, self.file_type)
            print(plt_name)
            plt.savefig(plt_name)

            f2, ax2 = plt.subplots()
            accuracy_vec = []
            for i in range(len(self.param_vals)):
                data = [round(float(f), 3) for f in accuracies[i]]
                accuracy_vec.append(self.epsilons)
                accuracy_vec.append(data)
                accuracy_vec.append(colors[i])
            ax2.plot(*accuracy_vec)
            plt.legend(["{}: {}".format(self.param_name, p) for p in self.param_vals])
            plt.title('Accuracy vs epsilon ({} = {})'.format(self.param_name, param))
            plt.xlabel('Epsilon')
            plt.ylabel('Accuracy')
            plt_name=  '{}/{}_{}_accuracy.{}'.format(self.transfer_plot_path, self.file_head, param, self.file_type)
            print(plt_name)
            plt.savefig(plt_name)
