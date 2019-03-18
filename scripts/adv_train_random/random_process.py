from adv_train_random import AdvTrainRandom
import tensorflow as tf
import cPickle
from tqdm import tqdm
from math import ceil
import numpy as np
from sklearn.metrics import accuracy_score
import csv
import parallel_utils

#python rf_scripts.py --random_process 1 --train_estimators 20

class RandomProcess:

	def __init__(self, config):

		random_data_path = config['random_data_path']
		n_test_adv_images = config['n_test_adv_images']
		boosting_iter = config['boosting_iter']
		random_estimators = config['random_estimators']
		no_threads = config['no_threads']
		epsilon_test_list = config['epsilons']

		mnist = tf.contrib.learn.datasets.load_dataset("mnist")
		eval_data = mnist.test.images  # Returns np.array
		eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
		indices = np.random.randint(0, len(eval_data), (n_test_adv_images,))


		eval_data_subset = eval_data[indices]
		eval_labels_subset = eval_labels[indices]


		clf_data_path = random_data_path + '/clfs/ab_est_{}_steps_{}.pkl'.format(random_estimators, boosting_iter)
		processed_data_path = random_data_path + '/processed_data'

		f = open(clf_data_path)
		ab = cPickle.load(f)
		f.close()

		estimators_list = ab.estimators_
		n_estimators = len(estimators_list)

		set_size = int(ceil(n_test_adv_images / no_threads))

		#_advGenParallel(self, lst, estimator, epsilon, rand_X, rand_y, index, set_size)

		#Generating adversarial images for best estimator (that last estimator) and calculating accuracy

		accuracy = []


		print "\n\n Calculating accuracies on test adversarial images"

		
		for index, est in enumerate(estimators_list):
			
			#if index % 5 == (len(estimators_list) - 1) % 5: #..., n_est - 11, n_est - 6, n_est - 1
			if index in [4, 24, 54, 79]: #CHANGE THIS

				print "\n\n Current Estimator %d" %index 
				cur_est_acc = []

				for epsilon_test in epsilon_test_list:

					#Generating some adversarial examples

					print "\n current epsilon %f" %epsilon_test

					# adv_examples_test_adv, adv_true_adv = ab._advGen(est, n_test_adv_images, n_test_adv_images, epsilon_test, eval_data_ss, eval_labels_ss)
					adv_examples_test_adv, true_labels_adv = parallel_utils.accumulate_parallel_function(ab._advGenParallel, n_test_adv_images, est, epsilon_test, eval_data_subset, eval_labels_subset, set_size)
					# adv_examples_test_adv, true_labels_adv = ab.randomGen(adv_examples_test_adv, true_labels_adv, epsilon_test)
					cur_est_acc.append(accuracy_score(true_labels_adv, est.predict(adv_examples_test_adv)))

				accuracy.append(cur_est_acc)


		# print "\n\n For the first estimator"

		# cur_est_acc = []

		# est = estimators_list[0]

		# for epsilon_test in epsilon_test_list:

		# 	#Generating some adversarial examples

		# 	print "\n current epsilon %f" %epsilon_test

		# 	adv_examples_test_adv, true_labels_adv = parallel_utils.accumulate_parallel_function(ab._advGenParallel, n_test_adv_images, est, epsilon_test, eval_data_subset, eval_labels_subset, set_size)
		# 	cur_est_acc.append(accuracy_score(true_labels_adv, est.predict(adv_examples_test_adv)))
	
		# accuracy.append(cur_est_acc)

		with open("{}/accuracy_{}.csv".format(processed_data_path, random_estimators), "wb") as f:
			writer = csv.writer(f)
			writer.writerows(accuracy)

		with open('{}/est_error_{}.csv'.format(processed_data_path, random_estimators), "wb") as f:
			writer  = csv.writer(f)
			writer.writerow(ab.estimator_errors_)


		#Writing the accuracies to a csv file

		print "\n\n\n Printing the accuracy 2 d array if file is not saved "
		print(accuracy)


		print "\n\n\n Printing the estimator error list if file is not saved "
		print(ab.estimator_errors_)


