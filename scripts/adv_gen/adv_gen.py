from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from scipy.optimize import minimize
from scipy.linalg import norm
import tensorflow as tf
import numpy as np
from scipy.optimize import basinhopping
import csv
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
import cPickle
import argparse
from sklearn.externals import joblib
import time
from multiprocessing import Pool
from multiprocessing import Process, Manager
import copy_reg
import types

def save_object(obj, filename):
   with open(filename, 'wb') as output:  # Overwrites any existing file.
       cPickle.dump(obj, output, cPickle.HIGHEST_PROTOCOL)

def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)


class AdversarialGeneration(object):

    def __init__(self, config):
        self.base_classifier = config['base_classifier']
        self.epsilons = config['epsilons']
        self.inner_iter = config['inner_iter']
        self.outer_iter = config['outer_iter']
        self.no_threads = config['no_threads']
        self.no_adv_images = config['no_adv_images']
        self.data_path = config['data_path']
        if 'train_size' not in config:
            self.train_size = 50000
        else:
            self.train_size = config['train_size']
        if self.base_classifier == 'random_forest':
            self.max_depth = config['max_depth']
            self.criterion = config['criterion']
            self.n_estimators = config['n_estimators']
        if self.base_classifier == 'svm':
            self.C_vals = config['C_vals']
    #Run multiple threads of adv_gen 
        self.main()
    
    def extractData(self):
        # Loading MNIST data
        # Load training and eval data
        mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        train_data = mnist.train.images  # Returns np.array
        train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
        eval_data = mnist.test.images  # Returns np.array
        eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

        #Subsampling the training dataset
        rand_idx_train = np.random.randint(0,len(train_data), ((self.train_size),))
        self.train_data_ss = train_data[rand_idx_train]
        self.train_labels_ss = train_labels[rand_idx_train]

        #Writing permutation of train data to a file
        file = '{}/permutations/perm_train.csv'.format(self.data_path)
        with open(file, 'w') as output:
            writer = csv.writer(output, delimiter = ',')
            writer.writerow(rand_idx_train)

        #Subsampling the testing dataset
        rand_idx_test = np.random.randint(0,len(eval_data), (5000,))
        self.eval_data_ss = eval_data[rand_idx_test]
        self.eval_labels_ss = eval_labels[rand_idx_test]

        #Writing permutation of test data to a file
        file = '{}/permutations/perm_test.csv'.format(self.data_path)
        with open(file, 'w') as output:
            writer = csv.writer(output, delimiter = ',')
            writer.writerow(rand_idx_test)
    
    def advGen_(self, param, epsilon, clf, no_adv_images):

        def fitness_func(noise, x, clf):
            x = np.array(x).reshape(1,-1)
            return -1*norm(clf.predict_proba(x) - clf.predict_proba(x+noise))

        t_start = time.time()

        """
        Adversarial image generation function for particular values of following parameters
        
        Input:
        ------
        est: no of estimators
        epsilon: epsilon value
        clfs: trained classifier for particular est, epsilon
        in_iter: no. of inner iterations for optimization
        out_iter: no. of outer iterations for optimization

        Output:
        -------
        image_vecs: vector of images for which adversarial images are calculated
        noise_vecs: vector of corresponding optimal noise
        correct_labels: true labels for those images

        """

        x0 = [0]*784
        

        '''
        CSV File contains:
        0. mean_fitness
        1. mean_noise
        2. var_noise, min_noise, max_noise, wrong_output, mean_l0,
        min_l0, max_l0, var_l0, mean_l1, min_l1, max_l1, var_l1
        '''

        print('Starting the optimization')

        
        correct_labels = []
        noise_vecs = []
        image_vecs = []


        print('Cur parameter: {}, Epsilon: {}'.format(param, epsilon))

        for image_no, image in enumerate(self.eval_data_ss[:no_adv_images,:]):
            x0 = [0] * 784
            print('Current Image: {}'.format(image_no))
            cons = ({'type': 'ineq',
                'fun': lambda noise: epsilon - norm(np.array(noise))})
            minimizer_kwargs = dict(method = "slsqp", args = (image,clf), constraints = cons, options = {'maxiter': self.inner_iter})
            res = basinhopping(fitness_func, niter = self.outer_iter, x0 = x0, minimizer_kwargs = minimizer_kwargs)
            image_vecs.append(image)
            noise_vecs.append(res['x'])
            correct_labels.append(self.eval_labels_ss[image_no])


        t_end = time.time()
        t_diff = t_end - t_start
        self.times["est_{}_eps_{}".format(param, epsilon)] = t_diff
        return image_vecs, noise_vecs, correct_labels

    def write_image_noise_labels_to_file(self, clf, epsilon, image_vecs, noise_vecs, correct_labels):

        base_param = self.base_classifier_params[0] # the classifier param that is being varied (Example: n_estimators for random forest)
        base_val = str(clf.get_params()[base_param])

        file = self.data_path + '/images' + '/' + base_param + '_' + base_val +\
                '_' + 'eps_' + str(epsilon) + '.csv'
        with open(file, 'w') as output:
            writer = csv.writer(output, delimiter=',')
            writer.writerows(image_vecs)
        file = self.data_path + '/noise' +'/'+ base_param  + '_' + base_val +\
                '_' + 'eps_' + str(epsilon) + '.csv'
        with open(file, 'w') as output:
            writer = csv.writer(output, delimiter = ',')
            writer.writerows(noise_vecs)

        file = self.data_path + '/correct_labels' +'/'+ base_param  + '_' + base_val +\
            '_' + 'eps_' + str(epsilon) + '.csv'
        with open(file, 'w') as output:
            writer = csv.writer(output, delimiter = ',')
            writer.writerow(correct_labels)
    
    def adv_gen_and_save(self, param, epsilon):
        print("Inside adv_gen_and_save")
        image_vecs, noise_vecs, correct_labels = self.advGen_(param, epsilon, self.clfs[param], self.no_adv_images)
        self.write_image_noise_labels_to_file(self.clfs[param], epsilon, image_vecs, noise_vecs, correct_labels)
        return 0

    def main(self):
        self.extractData()
        if self.base_classifier == 'random_forest':
            self.rf()
        if self.base_classifier == 'svm':
            self.svm()

    def rf(self):
        self.base_classifier_params = ['n_estimators', 'criterion', 'max_depth']
    
        #Training and saving the classifiers
        self.clfs = {}
        accuracy = []
        self.times = {}
        print(self.n_estimators)
        for est in self.n_estimators:
            print("EST:", est)
            params = {'n_estimators' : est,\
                        'criterion' : self.criterion,\
                        'max_depth' : self.max_depth}
            self.clfs[est] = RandomForestClassifier(**params)
            clf = self.clfs[est]
            clf.fit(self.train_data_ss, self.train_labels_ss)
            accuracy.append(accuracy_score(self.eval_labels_ss, clf.predict(self.eval_data_ss)))
            print("score: {}".format(accuracy_score(self.eval_labels_ss, clf.predict(self.eval_data_ss))))
        save_object(self.clfs, '{}/clfs.pkl'.format(self.data_path))
        print('saved object')
        tasks = []
        for est in self.n_estimators:
            for epsilon in self.epsilons:
                tasks.append((est, epsilon))
        print('created tasks')
        start_time = time.time()
        print("Starting the parallel script")
        Parallel(n_jobs=self.no_threads)(delayed(self.adv_gen_and_save)(est, eps) for (est, eps) in tasks)
        end_time = time.time()
        self.total_time = end_time - start_time
    def svm(self):
        self.base_classifier_params = ['C'] #PLEASE CHANGE THIS!!! (make another file_head instance variable so that the saved file has a lowercase c)
         #Training and saving the classifiers
        self.clfs = {}
        accuracy = []
        self.times = {}
        print(self.C_vals)
        for C in self.C_vals:
            print("C:", C)
            params = {'C': C,
                    'probability': True}
            self.clfs[C] = SVC(**params)
            clf = self.clfs[C]
            clf.fit(self.train_data_ss, self.train_labels_ss)
            accuracy.append(accuracy_score(self.eval_labels_ss, clf.predict(self.eval_data_ss)))
            print("score: {}".format(accuracy_score(self.eval_labels_ss, clf.predict(self.eval_data_ss))))
        save_object(self.clfs, '{}/clfs.pkl'.format(self.data_path))
        tasks = []
        for C in self.C_vals:
            for epsilon in self.epsilons:
                tasks.append((C, epsilon))
                start_time = time.time()
        Parallel(n_jobs=self.no_threads)(delayed(self.adv_gen_and_save)(C, eps) for (C, eps) in tasks)
        end_time = time.time()
        self.total_time = end_time - start_time
        

