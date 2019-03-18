from numpy import genfromtxt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class TrainPlot:
    def __init__(self, config):
        epsilons = config['epsilons']
        boosting_iter = config['boosting_iter']
        train_data_path = config['train_data_path']
        accuracy_file = train_data_path + '/processed_data/accuracy.csv'
        accuracies = genfromtxt(accuracy_file, delimiter= ',')

        est_to_acc = {}

        ax, fig = plt.subplots()
        plt.xlabel("Epsilon")
        plt.ylabel("Accuracy")
        plt.title("Adversarial Training: accuracy versus epsilon")
        print(epsilons)
        print(accuracies[-1])
        plt.plot(epsilons, accuracies[-1])
        plt_name = "{}/plots/accuracy.png".format(train_data_path)
        plt.savefig(plt_name)
        
