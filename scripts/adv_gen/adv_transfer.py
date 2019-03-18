#importing modules
import numpy as np
from numpy import genfromtxt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.linalg import norm
import csv
import cPickle

class AdversarialTransfer:

    def __init__(self, config):
        self.base_classifier = config['base_classifier']
        self.epsilons = config['epsilons']
        self.no_images = config['no_adv_images']
        self.raw_data_path = config['raw_data_path']
        self.transfer_data_path = config['transfer_data_path']
        if self.base_classifier == 'random_forest':
            self.param_name = 'Estimators'
            self.file_head = 'n_estimators'
            self.param_vals = config['n_estimators']
        elif self.base_classifier == 'svm':
            self.param_name = 'C'
            self.file_head = 'C'
            self.param_vals = config['C_vals']
        self.main()

    def main(self):

        # accuracy_list = []
        # fitness_list = []
        # std_fitness_list = []
        # l2_norm_list = []
        # std_l2_norm_list = []



        print("Opening classifiers")
        self.clfs = cPickle.load(open("{}/clfs.pkl".format(self.raw_data_path)))
        print("Opened classifiers")

        for param in self.param_vals:
            print("{}: {}".format(self.param_name, param))
            accuracies, fitnesses, std_fitnesses, l2_norms, std_l2_norms = self.process(param)
            
            #then write stuff to file

            file = '{}/{}_{}_accuracy.csv'.format(self.transfer_data_path, self.file_head, param)
            with open(file, 'w') as output:
                writer = csv.writer(output, delimiter = ',')
                writer.writerows(accuracies)
            file = '{}/{}_{}_fitness.csv'.format(self.transfer_data_path, self.file_head, param)
            with open(file, 'w') as output:
                writer = csv.writer(output, delimiter = ',')
                writer.writerows(fitnesses)
            file = '{}/{}_{}_std_fitness.csv'.format(self.transfer_data_path, self.file_head, param)
            with open(file, 'w') as output:
                writer = csv.writer(output, delimiter = ',')
                writer.writerows(std_fitnesses)
            file = '{}/{}_{}_l2.csv'.format(self.transfer_data_path, self.file_head, param)
            with open(file, 'w') as output:
                writer = csv.writer(output, delimiter = ',')
                writer.writerows(l2_norms)
            file = '{}/{}_{}_std_l2.csv'.format(self.transfer_data_path, self.file_head, param)
            with open(file, 'w') as output:
                writer = csv.writer(output, delimiter = ',')
                writer.writerows(std_l2_norms)

    def process(self, param): #param is the parameter value for the classifier for which the adversarial images are being tested on
        accuracies = []
        fitnesses = []
        std_fitnesses = []
        l2_norms = []
        std_l2_norms = []

        for adv_param in self.param_vals: #adv_param is the parameter value for adversarial images
            print("{}: {}".format(self.param_name, adv_param))
            acc_vec = []
            fitness_vec = []
            std_fitness_vec = []
            l2_vec = []
            std_l2_vec = []

            for eps in self.epsilons:
                print("Epsilon: {}".format(eps))
                clf = self.clfs[param]
                noises = genfromtxt("{}/noise/{}_{}_eps_{}.csv".format(self.raw_data_path, self.file_head, adv_param, eps), delimiter = ',')[:self.no_images]
                images = genfromtxt("{}/images/{}_{}_eps_{}.csv".format(self.raw_data_path, self.file_head, adv_param, eps), delimiter = ',')[:self.no_images]
                adv_images = images + noises
                true_labels = genfromtxt("{}/correct_labels/{}_{}_eps_{}.csv".format(self.raw_data_path, self.file_head, adv_param, eps), delimiter = ',')[:self.no_images] 
                non_adv_pred = clf.predict(images)
                adv_pred = clf.predict(adv_images)
                non_adv_probs = clf.predict_proba(images)
                adv_probs = clf.predict_proba(adv_images)

                accuracy = accuracy_score(adv_pred[:self.no_images], true_labels[:self.no_images])

                prob_perturbation = [norm(prob) for prob in (adv_probs - non_adv_probs)]
                l2 = [norm(noise) for noise in noises]

                mean_fitness = np.mean(prob_perturbation)
                std_fitness = np.std(prob_perturbation)
                mean_l2 = np.mean(np.array(l2))
                std_l2 = np.std(np.array(l2))
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
        return accuracies, fitnesses, std_fitnesses, l2_norms, std_l2_norms

