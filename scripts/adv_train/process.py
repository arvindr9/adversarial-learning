#importing modules
import numpy as np
from numpy import genfromtxt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.linalg import norm
import csv
import cPickle

class ProcessData:

    def init(self, config):
        self.base_classifier = config['base_classifier']
        self.epsilons = config['epsilons']
        self.no_images = config['no_adv_images']
        self.raw_data_path = config['raw_data_path']
        self.processed_data_path = config['processed_data_path']
        if self.base_classifier == 'random_forest':
            self.n_estimators = config['n_estimators']
        self.main()
    


    
    def main(self):
        accuracies = []
        fitnesses = []
        std_fitnesses = []
        l2_norms = []
        std_l2_norms = []


        print("Opening classifiers")
        clfs = cPickle.load(open("{}/clfs.pkl".format(self.raw_data_path)))
        print("Opened classifiers")

        if (self.base_classifier == 'random_forest_classifier'):
            accuracies, fitnesses, std_fitnesses, l2_norms, std_l2_norms = self.rf()
        elif (self.base_classifier == 'svm'):
            


        file = '{}/accuracy.csv'.format(self.processed_data_path)
        with open(file, 'w') as output:
            writer = csv.writer(output, delimiter = ',')
            writer.writerows(accuracies)
        file = '{}/fitness.csv'.format(self.processed_data_path)
        with open(file, 'w') as output:
            writer = csv.writer(output, delimiter = ',')
            writer.writerows(fitnesses)
        file = '{}/std_fitness.csv'.format(self.processed_data_path)
        with open(file, 'w') as output:
            writer = csv.writer(output, delimiter = ',')
            writer.writerows(std_fitnesses)
        file = '{}/l2.csv'.format(self.processed_data_path)
        with open(file, 'w') as output:
            writer = csv.writer(output, delimiter = ',')
            writer.writerows(l2_norms)
        file = '{}/std_l2.csv'.format(self.processed_data_path)
        with open(file, 'w') as output:
            writer = csv.writer(output, delimiter = ',')
            writer.writerows(std_l2_norms)

    def rf(self):
        accuracies = []
        fitnesses = []
        std_fitnesses = []
        l2_norms = []
        std_l2_norms = []

        for est in self.n_estimators:
            print("Estimator: {}".format(est))
            acc_vec = []
            fitness_vec = []
            std_fitness_vec = []
            l2_vec = []
            std_l2_vec = []

            for eps in self.epsilons:
                print("Epsilon: {}".format(eps))
                clf = clfs[est]
                noises = genfromtxt("{}/noise/n_estimators_{}_eps_{}.csv".format(self.raw_data_path, est, eps), delimiter = ',')[:self.no_images]
                images = genfromtxt("{}/images/n_estimators_{}_eps_{}.csv".format(self.raw_data_path, est, eps), delimiter = ',')[:self.no_images]
                adv_images = images + noises
                true_labels = genfromtxt("{}/correct_labels/n_estimators_{}_eps_{}.csv".format(self.raw_data_path, est, eps), delimiter = ',')[:self.no_images] 
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
        