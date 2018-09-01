#importing modules
from sklearn.ensemble import RandomForestClassifier
from scipy.optimize import minimize
from scipy.linalg import norm
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import basinhopping
import csv
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
import cPickle
import argparse
from sklearn.externals import joblib

#Loading MNIST data
# Load training and eval data
mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_data = mnist.train.images  # Returns np.array
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
eval_data = mnist.test.images  # Returns np.array
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

#Subsampling the training dataset
rand_idx = np.random.randint(0,len(train_data), (50000,))
train_data_ss = train_data[rand_idx]
train_labels_ss = train_labels[rand_idx]


#Subsampling the testing dataset
rand_idx = np.random.randint(0,len(eval_data), (5000,))
eval_data_ss = eval_data[rand_idx]
eval_labels_ss = eval_labels[rand_idx]




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
	return clf, accuracy_score(eval_labels_ss, clf.predict(eval_data_ss))






def advGen_(est, epsilon, clf, in_iter, out_iter, no_adv_images):

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


	print('Cur Estimator: {}, Epsilon: {}, inner iters : {}, outer iters : {}'.format(est, epsilon, in_iter, out_iter))

	for image_no, image in enumerate(eval_data_ss[:no_adv_images,:]):
		x0 = [0] * 784
		print('Current Image: {}'.format(image_no))
		cons = ({'type': 'ineq',
			'fun': lambda noise: epsilon - norm(np.array(noise))})
		minimizer_kwargs = dict(method = "slsqp", args = (image,clf), constraints = cons, options = {'maxiter': in_iter})
		res = basinhopping(fitness_func, niter = out_iter, x0 = x0, minimizer_kwargs = minimizer_kwargs)
		image_vecs.append(image)
		noise_vecs.append(res['x'])
		correct_labels.append(eval_labels_ss[image_no])



	return image_vecs, noise_vecs, correct_labels




def write_image_noise_labels_to_file(in_iter, out_iter, image_vecs, noise_vecs, correct_labels):

	file = heat_map_path + '/image_data' + '/' + list(base_classifier_params.keys())[0] + '_' + str(list(base_classifier_params.values())[0]) +\
			 '_' + 'eps_' + str(epsilon) + '_'+ 'in_'+ str(in_iter) + '_out_'+ str(out_iter) +'.csv'
	with open(file, 'w') as output:
		writer = csv.writer(output, delimiter=',')
		writer.writerows(image_vecs)
	file = heat_map_path + '/noise' +'/'+ list(base_classifier_params.keys())[0]  + '_' + str(list(base_classifier_params.values())[0]) +\
			 '_' + 'eps_' + str(epsilon) + '_'+ 'in_'+ str(in_iter) + '_out_'+ str(out_iter) +'.csv'
	with open(file, 'w') as output:
		writer = csv.writer(output, delimiter = ',')
		writer.writerows(noise_vecs)

	file = heat_map_path + '/true_labels' +'/'+ list(base_classifier_params.keys())[0]  + '_' + str(list(base_classifier_params.values())[0]) +\
		  '_' + 'eps_' + str(epsilon) + '_'+ 'in_'+ str(in_iter) + '_out_'+ str(out_iter) +'.csv'
	with open(file, 'w') as output:
		writer = csv.writer(output, delimiter = ',')
		writer.writerow(correct_labels)



def adv_gen_and_save(in_iter, out_iter):

	image_vecs, noise_vecs, correct_labels = advGen_(n_estimators, epsilon, clf, in_iter, out_iter, no_adv_images)
	write_image_noise_labels_to_file(in_iter, out_iter, image_vecs, noise_vecs, correct_labels)


def main():

	
	parser = argparse.ArgumentParser()
	parser.add_argument("--heat_map_path", default = "heatmap_data", help="where all the data and files related to heat map gen are stored")
	parser.add_argument("--base_estimator", default = "random_forest", help="base estimator {'random_forest'}")
	parser.add_argument("--n_estimators", default = 20, help ="no. of estimators in base estimators", type =int)
	parser.add_argument("--criterion", default = 'entropy', help ="criterion for base estimator")
	parser.add_argument("--max_depth", default = 10, help = "maximum depth for base estimator", type = int)
	parser.add_argument("--epsilon", default = 0.5, help = "epsilon value for optimization", type = int)
	parser.add_argument("--no_adv_images", default = 50, help = "number of adversarial to be generated for optimization", type = int)
	parser.add_argument("--no_of_threads", default = 8, help = "number of threads to run in parallel", type = int)
	args = parser.parse_args()
	arguments = args.__dict__

	#Make a dictionary of all arguments
	args_dict = {k: v for k,v in arguments.items()}

	#Declaring all the variables global
	global heat_map_path
	global base_estimator
	global n_estimators
	global criterion
	global max_depth
	global epsilon
	global no_adv_images
	global base_classifier_params
	global base_classifier
	global clf
	global no_jobs


	heat_map_path = args_dict['heat_map_path']
	base_estimator = args_dict['base_estimator']
	n_estimators = args_dict['n_estimators']
	criterion = args_dict['criterion']
	max_depth = args_dict['max_depth']
	epsilon = args_dict['epsilon']
	no_adv_images = args_dict['no_adv_images']
	no_jobs = args_dict['no_of_threads']

	base_classifier_params = {'n_estimators' : n_estimators,\
						  'criterion' : criterion,\
						  'max_depth' : max_depth}

	if(base_estimator == "random_forest"):
		base_classifier = RandomForestClassifier(**base_classifier_params)

	

	#Pretraining the classifier
	clf, _ = pre_training()

	joblib.dump(clf, './heat_map_trained_classifier.pkl') 

	inner_iter = [5, 10, 20, 50, 100]
	outer_iter = [1, 2, 5, 10]

	tasks = []


	for in_iter in inner_iter:
		for out_iter in outer_iter:

			tasks.append((in_iter, out_iter))

	
	Parallel(n_jobs=no_jobs)(delayed(adv_gen_and_save)(in_iter, out_iter) for (in_iter, out_iter) in tasks)


if __name__ == '__main__':

	
	main()
	

