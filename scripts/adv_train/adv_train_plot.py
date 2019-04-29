from numpy import genfromtxt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class TrainPlot:
    def __init__(self, config):
        train_size = config['train_size']
        epsilons = config['epsilons']
        boosting_iter = config['boosting_iter']
        train_data_path = config['train_data_path']
        gen_data_path = config['gen_data_path']
        train_estimators = config['train_estimators']
        gen_estimators = config['gen_estimators'] #list of estimators used in adv_gen
        accuracy_file = train_data_path + '/processed_data/accuracy_{}'.format(train_estimators) + '_train_size_' + str(train_size) + '.csv'
        accuracies = genfromtxt(accuracy_file, delimiter= ',')
        vanilla_accuracy_file = gen_data_path + '/accuracy.csv'
        vanilla_accuracies = genfromtxt(vanilla_accuracy_file, delimiter = ',')[gen_estimators.index(train_estimators)]

        steps = [4, 29, 54, 79] #make a way to compute steps?

        # estimator_values = [boosting_iter - 1]
        # i = boosting_iter - 6
        # while i >= 5:
        #     i -= 5
        #     estimator_values = [i] + estimator_values
        # print(estimator_values)

        est_to_acc = {}

        ax, fig = plt.subplots()
        plt.xlabel("Epsilon")
        plt.ylabel("Accuracy")
        plt.title("Adversarial Training: accuracy versus epsilon")
        print(epsilons)
        print(accuracies[-1])
        #plt.plot(epsilons, accuracies[-1])
        '''
        Also include accuracy data from advgen
        '''

        # colors = ['r-', 'b-', 'g-', 'y-', 'p-', 'k-', 'bo-', 'c-']

        # for i in range(len(estimator_values)):
        #     if i % 5 == 0:
        #         plt.plot(epsilons, accuracies[i])
        f, ax = plt.subplots()
        for i in range(len(steps)):
            plt.plot(epsilons, accuracies[i])
        plt.plot(epsilons, vanilla_accuracies)
        legend = ["Steps:{}".format(step) for step in steps] + ["Vanilla-trained classifier"]
        # legend = ["Estimators:{}".format(estimator_values[i]) for i in range(0, len(estimator_values), 5)] + ["Vanilla-trained classifier"]
        plt.legend(legend)
        plt.title("Adversarial Training")
        plt_name = "{}/plots/accuracy_train_est_{}".format(train_data_path, train_estimators) + '_train_size_' + str(train_size) + '.png'
        plt.savefig(plt_name)
        
