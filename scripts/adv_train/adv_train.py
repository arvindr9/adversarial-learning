import numpy as np
import csv
import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.base import clone
from scipy.optimize import basinhopping
from scipy.linalg import norm
import time
import cPickle
from multiprocessing import Process, Manager
import os
from math import ceil
import parallel_utils
import copy

def save_object(obj, filename):
   with open(filename, 'wb') as output:  # Overwrites any existing file.
       cPickle.dump(obj, output, cPickle.HIGHEST_PROTOCOL)
#

class AdversarialBoost:

	"""
	Class for Adversarial Boosting
	Parameters
	-----------
	base_estimator : scikit learn classifier
	n_estimators : number of estimators in boosting loop
	epsilon : (float) radius of noise for adversarial examples generation
	n_boost_random_train_samples : number of random examples to take from training data for generating adversarial examples
	estimator_params : estimator parameters
	"""

	def __init__(self, config):

		self.base_classifier = config['base_classifier']
		if self.base_classifier == 'random_forest':
			self.base_estimator = RandomForestClassifier()
		elif self.base_classifier == 'svm':
			self.base_estimator = SVC()

		#parameters for the training loop
		self.inner_iter = config['inner_iter']
		self.outer_iter = config['outer_iter']
		self.epsilon = config['epsilon']

		self.n_boosting_clf = config['boosting_iter']
		self.estimators_ = []
		self.estimator_errors_ = np.ones(self.n_boosting_clf, dtype= np.float64)
		
		self.n_boost_random_train_samples = config['n_boost_random_train_samples']
		self.estimator_params = config['estimator_params']

		self.train_size = config['train_size']
		self.train_data_path = config['train_data_path']
		self.no_threads = config['no_threads']

		if(isinstance(self.base_estimator, SVC)):
				
			self.base_estimator.set_params(**{'probability': True})

		if(isinstance(self.base_estimator, RandomForestClassifier)):

			self.base_estimator.set_params(**{'max_depth': 5})

		if not hasattr(self.base_estimator, 'predict_proba'):
			raise TypeError("Base estimator has no attribute 'predict_proba'")


	def fit(self, X, y):

		clean_indices = np.random.randint(0, len(X), (self.train_size,))

		adv_data = np.array([])
		adv_labels = np.array([])

		clean_data = X[clean_indices]
		clean_labels = y[clean_indices]

		#each iteration: take 500 random examples, generate adv and append to the adv list

		for iboost in range(self.n_boosting_clf):

			clean_data = X[clean_indices]
			clean_labels = y[clean_indices]

			X_union = None
			y_union = None
			if np.any(adv_data.shape): #check whether the adv list has elements
				X_union = np.concatenate((clean_data, adv_data), axis = 0)
				y_union = np.concatenate((clean_labels, adv_labels), axis = None)
			else:
				X_union = clean_data
				y_union = clean_labels

			print("Boosting step : {}".format(iboost))

			# Training the weak classifier
			cur_estimator, estimator_error = self._boost(iboost, X_union, y_union)

			#Generating adversarial examples
			print("Generating adversarial examples.....")

			rand_idx = np.random.randint(0,len(clean_data), (self.n_boost_random_train_samples,))
			rand_X = clean_data[rand_idx] #sample images from the clean data to append to the adv set
			rand_y = clean_labels[rand_idx].reshape(-1, 1)
			
			'''
			set_size: the number of adversarial images to generate in each thread
			indexArray: contains the index of the adversarial image to be generated first in each thread
			ListAdvImages: contains the adversarial examples and the class of each adversarial example
			'''
			set_size = int(ceil(self.n_boost_random_train_samples / self.no_threads))
			print("adv images:", self.n_boost_random_train_samples)
			print("no_threads: ", self.no_threads)
			print("set_size:", set_size)
			
			adv_examples, adv_y = parallel_utils.accumulate_parallel_function(self._advGenParallel, self.n_boost_random_train_samples, cur_estimator, self.epsilon, rand_X, rand_y, set_size)

			print("Done with Generating adversarial examples.....") 

			b_adv_conc_start = time.time()

			
			#Append the new adversarial examples to the list of adversarial examples
			if np.any(adv_data.shape):
				adv_data = np.concatenate((adv_data, adv_examples), axis = 0)
				adv_labels = np.concatenate((adv_labels, adv_y), axis = None)
			else:
				adv_data = adv_examples
				adv_labels = adv_y

			self.estimator_errors_[iboost] = estimator_error

			if iboost % 5 == 0:
				if iboost >= 5:
					os.remove(self.train_data_path + '/clfs/ab_parallel_' + str(self.estimator_params['n_estimators']) +'_'+ str(iboost - 5) + '_train_size_' + str(self.train_size) + '.pkl')
				save_object(self, self.train_data_path + '/clfs/ab_parallel_' + str(self.estimator_params['n_estimators']) +'_'+ str(iboost) + '_train_size_' + str(self.train_size) + '.pkl')

		return self


	def _boost(self, iboost, X, y):

		estimator = self._make_estimator()
		estimator.fit(X,y)
		self.estimators_.append(estimator)

		y_predict = estimator.predict(X)

		if iboost == 0:
			self.classes_ = getattr(estimator, 'classes_', None)
			self.n_classes_ = len(self.classes_)

		# Instances incorrectly classified
		incorrect = y_predict != y

		# Error fraction
		estimator_error = np.mean(np.average(incorrect, axis=0)) ####

		return estimator, estimator_error


	def _make_estimator(self):

		estimator = clone(self.base_estimator)

		if(self.estimator_params is not None):
			estimator.set_params(**dict(self.estimator_params))

		return estimator


	def fitness_func(self, noise, x, clf):

		x = np.array(x).reshape(1,-1)
		return -1*norm(clf.predict_proba(x) - clf.predict_proba(x+noise))


	def _advGenParallel(self, lst, estimator, epsilon, rand_X, rand_y, index, set_size, random = False):

		if random == True:
			self.inner_iter = 1
			self.outer_iter = 1

		x0 = [0] * np.shape(rand_X)[1]

		cons = ({'type': 'ineq', 'fun': lambda noise: epsilon - norm(np.array(noise))})

		optimal_noise = []
		
		X = None
		y = None
		if index + set_size >= rand_X.shape[0]:
			X = rand_X[index:,:]
			y = rand_y[index:]
		else:
			X = rand_X[index:index + set_size, :]
			y = rand_y[index:index + set_size]

		for image_no, image in enumerate(X):
			minimizer_kwargs = dict(method = "slsqp", args = (image, estimator), constraints = cons, options = {'maxiter': self.inner_iter})
			res = basinhopping(self.fitness_func, niter = self.outer_iter, x0 = x0, minimizer_kwargs = minimizer_kwargs)
			cur_noise = res['x']
			optimal_noise.append(cur_noise)

		lst.append((X + optimal_noise, y)) #(adv. images, labels) #changed from y[:][0]

	def main(self):
		#Loading Mnist data
		# Load training and eval data
		mnist = tf.contrib.learn.datasets.load_dataset("mnist")
		train_data = mnist.train.images  # Returns np.array
		train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
		eval_data = mnist.test.images  # Returns np.array
		eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

		#Shuffling the training dataset
		rand_idx_train = np.random.randint(0,len(train_data), (50000,)) 
		train_data = train_data[rand_idx_train]
		train_labels = train_labels[rand_idx_train]

		#Shuffling the testing dataset
		rand_idx_test = np.random.randint(0,len(eval_data), (50000,))
		eval_data = eval_data[rand_idx_test]
		eval_labels = eval_labels[rand_idx_test]

		#Adversarial Training
		print "\n\n Training Random forest with adversarial images"
		self.fit(train_data, train_labels)

		# Saving the random forest to a pkl file
		save_object(self, self.train_data_path + '/clfs/ab_est_' + str(self.estimator_params['n_estimators']) +'_steps_'+ str(self.n_boosting_clf) + '_train_size_' + str(self.train_size) + '.pkl')