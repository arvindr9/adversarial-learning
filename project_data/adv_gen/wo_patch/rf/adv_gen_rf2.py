#importing modules
from sklearn.ensemble import RandomForestClassifier
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

#Loading MNIST data
# Load training and eval data
mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_data = mnist.train.images  # Returns np.array
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
eval_data = mnist.test.images  # Returns np.array
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

#Subsampling the training dataset
rand_idx_train = np.random.randint(0,len(train_data), (50000,))
train_data_ss = train_data[rand_idx_train]
train_labels_ss = train_labels[rand_idx_train]

#Writing permutation of train data to a file
file = 'raw_data/permutations/perm_train.csv'
with open(file, 'w') as output:
	writer = csv.writer(output, delimiter = ',')
	writer.writerow(rand_idx_train)

#Subsampling the testing dataset
rand_idx_test = np.random.randint(0,len(eval_data), (5000,))
eval_data_ss = eval_data[rand_idx_test]
eval_labels_ss = eval_labels[rand_idx_test]

#Writing permutation of test data to a file
file = 'raw_data/permutations/perm_test.csv'
with open(file, 'w') as output:
	writer = csv.writer(output, delimiter = ',')
	writer.writerow(rand_idx_test)

#Fitness function
def fitness_func(noise, x, clf):

	x = np.array(x).reshape(1,-1)
	return -1*norm(clf.predict_proba(x) - clf.predict_proba(x+noise))


def pre_training():

	"""
	pre training of the base_classifier

	Input:
	------
	base_classifier: sklearn classifier (default = RandomForestClassifier())
	base_classifier_params: dictionary of base classifier parameters (default (random forest parameters) = {'n_estimators' = 10, 'criterion' = 'entropy', 'max_depth' =10})

	"""
	#Pretraining each classifier

	clf = base_classifier
	clf.fit(train_data_ss, train_labels_ss)
	return clfs

def advGen_(est, epsilon, clf, no_adv_images):
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


	print('Cur Estimator: {}, Epsilon: {}'.format(est, epsilon))

	for image_no, image in enumerate(eval_data_ss[:no_adv_images,:]):
		x0 = [0] * 784
		print('Current Image: {}'.format(image_no))
		cons = ({'type': 'ineq',
			'fun': lambda noise: epsilon - norm(np.array(noise))})
		minimizer_kwargs = dict(method = "slsqp", args = (image,clf), constraints = cons, options = {'maxiter': inner_iter})
		res = basinhopping(fitness_func, niter = outer_iter, x0 = x0, minimizer_kwargs = minimizer_kwargs)
		image_vecs.append(image)
		noise_vecs.append(res['x'])
		correct_labels.append(eval_labels_ss[image_no])


	t_end = time.time()
	t_diff = t_end - t_start
	times["est_{}_eps_{}".format(est, epsilon)] = t_diff
	return image_vecs, noise_vecs, correct_labels

def write_image_noise_labels_to_file(clf, epsilon, image_vecs, noise_vecs, correct_labels):

	base_param = base_classifier_params[0]
	base_val = str(clf.get_params()[base_param])

	file = data_path + '/images' + '/' + base_param + '_' + base_val +\
			 '_' + 'eps_' + str(epsilon) + '.csv'
	with open(file, 'w') as output:
		writer = csv.writer(output, delimiter=',')
		writer.writerows(image_vecs)
	file = data_path + '/noise' +'/'+ base_param  + '_' + base_val +\
			 '_' + 'eps_' + str(epsilon) + '.csv'
	with open(file, 'w') as output:
		writer = csv.writer(output, delimiter = ',')
		writer.writerows(noise_vecs)

	file = data_path + '/correct_labels' +'/'+ base_param  + '_' + base_val +\
		  '_' + 'eps_' + str(epsilon) + '.csv'
	with open(file, 'w') as output:
		writer = csv.writer(output, delimiter = ',')
		writer.writerow(correct_labels)

def adv_gen_and_save(n_estimators, epsilon):
	image_vecs, noise_vecs, correct_labels = advGen_(n_estimators, epsilon, clfs[n_estimators], no_adv_images)
	write_image_noise_labels_to_file(clfs[n_estimators], epsilon, image_vecs, noise_vecs, correct_labels)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_path", default = "raw_data", help="where all the data and files related to heat map gen are stored")
	parser.add_argument("--base_estimator", default = "random_forest", help="base estimator {'random_forest'}")
	parser.add_argument("--n_estimators", default = [1, 2, 5, 10, 20, 50, 100], help ="no. of base estimators", type = list)
	parser.add_argument("--criterion", default = 'entropy', help ="criterion for base estimator")
	parser.add_argument("--max_depth", default = 10, help = "maximum depth for base estimator", type = int)
	parser.add_argument("--epsilons", default = [0., 0.001, 0.01, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0], help = "epsilon values for optimization", type = list) #CHANGE
	parser.add_argument("--no_adv_images", default = 3, help = "number of adversarial to be generated for optimization", type = int)
	parser.add_argument("--no_of_threads", default = 70, help = "number of threads to run in parallel", type = int)
	parser.add_argument("--inner_iter", default = 20, help = "number of iterations of slsqp in the inner loop of the optimization", type = int)
	parser.add_argument("--outer_iter", default = 5, help = "Number of iterations of basinhopping for the optimization")
	args = parser.parse_args()
	arguments = args.__dict__

	#Make a dictionary of all arguments
	args_dict = {k: v for k,v in arguments.items()}

	#Declaring all the variables global
	global data_path
	global base_estimator
	global n_estimators
	global criterion
	global max_depth
	global epsilons
	global no_adv_images
	global base_classifier_params
	global base_classifier
	global clfs
	global no_jobs
	global inner_iter
	global outer_iter
	global times

	data_path = args_dict["data_path"]
	base_estimator = args_dict["base_estimator"]
	n_estimators = args_dict["n_estimators"]
	criterion = args_dict["criterion"]
	max_depth = args_dict["max_depth"]
	n_estimators = args_dict["n_estimators"]
	criterion = args_dict["criterion"]
	max_depth = args_dict["max_depth"]
	epsilons = args_dict["epsilons"]
	no_adv_images = args_dict["no_adv_images"]
	no_jobs = args_dict["no_of_threads"]
	inner_iter = args_dict["inner_iter"]
	outer_iter = args_dict["outer_iter"]


	if(base_estimator == "random_forest"):

		base_classifier_params = ['n_estimators', 'criterion', 'max_depth']
		
			#Training and saving the classifiers
		clfs = {}
		accuracy = []
		times = {}
		for est in n_estimators:
			params = {'n_estimators' : n_estimators,\
						  'criterion' : criterion,\
						  'max_depth' : max_depth}
			base_classifier = RandomForestClassifier(**params)
			clfs[est] = RandomForestClassifier(n_estimators = est, criterion = 'entropy', max_depth =10)
			clf = clfs[est]
			clf.fit(train_data_ss, train_labels_ss)
			accuracy.append(accuracy_score(eval_labels_ss, clf.predict(eval_data_ss)))
			print("score: {}".format(accuracy_score(eval_labels_ss, clf.predict(eval_data_ss))))
		joblib.dump(clfs, 'raw_data/clfs.pkl')
		tasks = []
		for est in n_estimators:
			for epsilon in epsilons:
				tasks.append((est, epsilon))

		Parallel(n_jobs=no_jobs)(delayed(adv_gen_and_save)(est, eps) for (est, eps) in tasks)

if __name__ == '__main__':
    main()