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

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        cPickle.dump(obj, output, cPickle.HIGHEST_PROTOCOL)


class AdversarialBoost:

	"""
	Class for Adversarial Boosting


	Parameters
	-----------
	base_estimator : scikit learn classifier
	n_estimators : number of estimators in boosting loop
	epsilon : (float) radius of noise for adversarial examples generation
	n_adv : (int) number of adversarial examples to train in each loop
	n_boost_random_train_samples : number of random examples to take from training data for generating adversarial examples
	estimator_params : estimator parameters

	"""

	def __init__(self, base_estimator, n_estimators, epsilon, n_adv, n_boost_random_train_samples, estimator_params = None):


		self.base_estimator = base_estimator
		self.n_estimators = n_estimators
		self.estimators_ = []
		self.estimator_errors_ = np.ones(n_estimators, dtype= np.float64)
		self.epsilon = epsilon
		self.n_adv = n_adv
		self.n_boost_random_train_samples = n_boost_random_train_samples
		self.estimator_params = estimator_params
		self.total_boosting_time = [] #Total time for each boosting step
		self.fitting_time = []	#Time for training one boosting classifier
		self.adv_generation_time = [] #Total time for generating all the adversarial images in each step
		self.concatenate_time = [] #Time for concatenating images
		self.per_image_advGen_time = [] #Time per image adversarial image generation


		if(isinstance(self.base_estimator, SVC)):
				
			self.base_estimator.set_params(**{'probability': True})

		if(isinstance(self.base_estimator, DecisionTreeClassifier)):

			self.base_estimator.set_params(**{'max_depth': 10})

		if(isinstance(self.base_estimator, RandomForestClassifier)):

			self.base_estimator.set_params(**{'max_depth': 10})

		if not hasattr(self.base_estimator, 'predict_proba'):
			raise TypeError("Base estimator has no attribute 'predict_proba'")


	def fit(self, X, y):

		for iboost in range(self.n_estimators):

			b_tot_start = time.time()


			b_fit_start = time.time()


			print "Boosting step : %d" %iboost


			# Training the weak classifier
			cur_estimator, estimator_error = self._boost(iboost, X, y)


			b_fit_end = time.time()

			#Generating adversarial examples

			b_adv_gen_start = time.time()

			print "Generating adversarial examples....." 


			adv_examples, adv_y = self._advGen(cur_estimator, self.n_adv, self.n_boost_random_train_samples, self.epsilon, X, y)


			b_adv_gen_end = time.time()


			print "\nDone with Generating adversarial examples....." 

			b_adv_conc_start = time.time()

			
			#Taking a union of adversarial examples and original training data
			X = np.concatenate((X, adv_examples), axis = 0)
			y = np.concatenate((y, adv_y), axis = 0)


			b_adv_conc_end = time.time()


			self.estimator_errors_[iboost] = estimator_error

			# Stop if error is zero
			# if estimator_error == 0:

			# 	print "Terminated after %d boosting steps since estimator_error = 0 \n" %(iboost+1)
			# 	break

			print "\n"

			b_tot_end = time.time()

			self.total_boosting_time.append(b_tot_end - b_tot_start)

			self.fitting_time.append(b_fit_end - b_fit_start)

			self.adv_generation_time.append(b_adv_gen_end - b_adv_gen_start)

			self.concatenate_time.append(b_adv_conc_end - b_adv_conc_start)

		return self



	def _boost(self, iboost, X, y):

			

		estimator = self._make_estimator()


		# estimator.fit(X, y, sample_weight=sample_weight)
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


		# Stop if classification is perfect
		# if estimator_error <= 0:

		# 	return estimator, 0                    ####

		n_classes = self.n_classes_


		# Stop if the error is at least as bad as random guessing
		# if estimator_error >= 1. - (1. / n_classes):

		# 	self.estimators_.pop(-1)
		# 	if len(self.estimators_) == 0:

		# 		raise ValueError('BaseClassifier in Adversarial Boost classifier '
		# 						 'ensemble is worse than random, ensemble '
		# 						 'can not be fit.')
		# 		return None, None                       ####

		
		return estimator, estimator_error


	def _make_estimator(self):

		estimator = clone(self.base_estimator)

		if(self.estimator_params is not None):

			estimator.set_params(**dict(self.estimator_params))

		return estimator


	def fitness_func(self, noise, x, clf):

		x = np.array(x).reshape(1,-1)
		return -1*norm(clf.predict_proba(x) - clf.predict_proba(x+noise))


	def _advGen(self, estimator, n_adv, n_boost_random_train_samples, epsilon, X, y):


		x0 = [0] * np.shape(X)[1]

		cons = ({'type': 'ineq', 'fun': lambda noise: epsilon - norm(np.array(noise))})

		optimal_noise = []
		optimal_fitness = []

		rand_idx = np.random.randint(0,len(X), (n_boost_random_train_samples,))
		rand_X = X[rand_idx]
		rand_y = y[rand_idx]

		for image_no, image in enumerate(rand_X):
			
			per_image_advGen_start = time.time()

			print "Generating adversarial examples for image number : %d\n" %image_no ,

			minimizer_kwargs = dict(method = "slsqp", args = (image, estimator), constraints = cons, options = {'maxiter': 100})
			res = basinhopping(self.fitness_func, niter = 10, x0 = x0, minimizer_kwargs = minimizer_kwargs)
			optimal_fitness.append(res['fun'])
			cur_noise = res['x']

			per_image_advGen_end = time.time()

			self.per_image_advGen_time.append(per_image_advGen_end - per_image_advGen_start)
			
			optimal_noise.append(cur_noise)

		index_noise = sorted(range(len(optimal_fitness)), key=lambda k: optimal_fitness[k])

		adv_examples = []
		adv_y = []
		for index in index_noise[:n_adv]:
			adv_examples.append(rand_X[index, :] + optimal_noise[index])
			adv_y.append(rand_y[index])

		return np.array(adv_examples), adv_y

		

if(__name__ == '__main__'):


	clf = 'rf'

	estimator_params = {'n_estimators': 20, 'criterion': 'entropy', 'max_depth': 10, 'min_samples_split': 5}
	#Loading Mnist data
	# Load training and eval data
	mnist = tf.contrib.learn.datasets.load_dataset("mnist")
	train_data = mnist.train.images  # Returns np.array
	train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
	eval_data = mnist.test.images  # Returns np.array
	eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

	#Subsampling the training dataset
	rand_idx = np.random.randint(0,len(train_data), (500,))
	train_data_ss = train_data[rand_idx]
	train_labels_ss = train_labels[rand_idx]

	#Subsampling the testing dataset
	rand_idx = np.random.randint(0,len(eval_data), (500,))
	eval_data_ss = eval_data[rand_idx]
	eval_labels_ss = eval_labels[rand_idx]

	#Training parameters
	no_boosting_clf = 100
	epsilon_train = 0.3
	n_advimages = 100
	n_boost_random_train_samples =  100

	#Testing parameters
	n_test_adv_images = 200
	epsilon_test_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
	#Adversarial Training

	print "\n\n Training Random forest with adversarial images"

	ab = AdversarialBoost(RandomForestClassifier(), no_boosting_clf, epsilon_train, n_advimages, n_boost_random_train_samples, estimator_params = estimator_params)
	ab.fit(train_data_ss, train_labels_ss)

	#Saving the random forest to a pkl file
	save_object(ab, 'ab_' + str(no_boosting_clf) + '.pkl')

	#Saving the data for time benchmarks
		# 	self.total_boosting_time = [] #Total time for each boosting step
		# self.fitting_time = []	#Time for training one boosting classifier
		# self.adv_generation_time = [] #Total time for generating all the adversarial images in each step
		# self.concatenate_time = [] #Time for concatenating images
		# self.per_image_advGen_time
	with open('boosting_time_' + str(no_boosting_clf) + '.csv', "wb") as f:
		writer = csv.writer(f)
		writer.writerow(ab.boosting_time)
	with open('fitting_time_' + str(no_boosting_clf) + '.csv', "wb") as f:
		writer = csv.writer(f)
		writer.writerow(ab.fitting_time)
	with open('adv_generation_time_' + str(no_boosting_clf) + '.csv', "wb") as f:
		writer = csv.writer(f)
		writer.writerow(ab.adv_generation_time)
	with open('concatenate_time_' + str(no_boosting_clf) + '.csv', "wb") as f:
		writer = csv.writer(f)
		writer.writerow(ab.concatenate_time)
	with open('per_image_advGen_time_' + str(no_boosting_clf) + '.csv', "wb") as f:
		writer = csv.writer(f)
		writer.writerow(ab.per_image_advGen_time)
		



	estimators_list = ab.estimators_

	#Generating adversarial images for best estimator (that last estimator) and calculating accuracy


	print "\n\n Calculating accuracies on test adversarial images"

	
	# for index, est in enumerate(estimators_list):

	# 	print "\n\n Current Estimator %d" %index 
	# 	cur_est_acc = []

		# for epsilon_test in epsilon_test_list:

		# 	#Generating some adversarial examples

		# 	print "\n current epsilon %f" %epsilon_test

		# 	adv_examples_test_adv, adv_true_adv = ab._advGen(est, n_test_adv_images, n_test_adv_images, epsilon_test, eval_data_ss, eval_labels_ss)
		# 	cur_est_acc.append(accuracy_score(adv_true_adv, est.predict(adv_examples_test_adv)))

	# 	accuracy.append(cur_est_acc)

	accuracy = []
	
	print "\n\n For last estimator"

	cur_est_acc = []

	est = estimators_list[-1]

	for epsilon_test in epsilon_test_list:

		#Generating some adversarial examples

		print "\n current epsilon %f" %epsilon_test

		adv_examples_test_adv, adv_true_adv = ab._advGen(est, n_test_adv_images, n_test_adv_images, epsilon_test, eval_data_ss, eval_labels_ss)
		cur_est_acc.append(accuracy_score(adv_true_adv, est.predict(adv_examples_test_adv)))
  
	accuracy.append(cur_est_acc)

	print "\n\n For the first estimator"

	cur_est_acc = []

	est = estimators_list[0]

	for epsilon_test in epsilon_test_list:

		#Generating some adversarial examples

		print "\n current epsilon %f" %epsilon_test

		adv_examples_test_adv, adv_true_adv = ab._advGen(est, n_test_adv_images, n_test_adv_images, epsilon_test, eval_data_ss, eval_labels_ss)
		cur_est_acc.append(accuracy_score(adv_true_adv, est.predict(adv_examples_test_adv)))
  
	accuracy.append(cur_est_acc)


	#Writing the accuracies to a csv file

	print "\n\n\n Printing the accuracy 2 d array if file is not saved "
	print(accuracy)


	print "\n\n\n Printing the estimator error list if file is not saved "
	print(ab.estimator_errors_)

	with open("accuracy.csv", "wb") as f:
	    writer = csv.writer(f)
	    writer.writerows(accuracy)

	with open('est_error.csv', "wb") as f:
		writer  = csv.writer(f)
		writer.writerow(ab.estimator_errors_)






	
