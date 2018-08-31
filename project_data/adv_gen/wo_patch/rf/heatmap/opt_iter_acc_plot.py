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


#Global variables
#Base estimator parameters
base_estimator = RandomForestClassifier()
base_classifier_params = {'n_estimators' = 20,
						  'criterion' = 'entropy',
						  'max_depth' = 10}

#No iterative Optimization parameters
epsilon = 0.5
no_adv_images = 50

#Data path
heat_map_path = ''


def calc_accuracy_and_display(est, epsilon, clf, in_iter, out_iter, no_adv_images, image_vecs, noise_vecs, correct_labels):

	"""
	Calculate the accuracy metrics for particular value of following parameters

	Input:
	------
	est: no of estimators
	epsilon: epsilon value
	clf: trained classifier for particular estimator and epsilon
	in_iter: no. of inner iterations for optimization
	out_iter: no. of outer iterations for optimization
	image_vecs: vector of images for which adversarial images are calculated
	noise_vecs: vector of corresponding optimal noise
	correct_labels: true labels for those images

	Output:
	-------
	acc_normal: accuracy of random forest on normal images
	acc_adv_images: accuracy of random forest on adversarial images
	misclassification: misclassification error
	"""
	



	normal_image_label = clf.predict(eval_data_ss)
	acc_normal = accuracy_score(eval_labels_ss, normal_image_label)

	adv_image_label = []
	for image, noise in zip(image_vecs,noise_vecs):
		adv_image_label.append(clf.predict((image+noise).reshape(1,-1))[0])
	adv_image_label = np.array(adv_image_label)
	acc_adv_images = accuracy_score(eval_labels_ss, adv_image_label)

	misclassification = (len(normal_image_label[normal_image_label!=adv_image_label])/len(normal_image_label))*100

	return acc_normal, acc_adv_images, misclassification


def read_image_noise_labels_from_file():

	

def plot_heatmap():



def main():

	parser = argparse.ArgumentParser()
    parser.add_argument("--heat_map_path", default="/home/sagarwal311/Adversarial-Learning/heatmap", help="where all the data and files related to heat map gen are stored")
    parser.add_argument("--base_estimator", default="random_forest", help="base estimator {'random_forest'}")
    dparser.add_argument("--n_estimators", default=20, help ="no. of estimators in base estimators", type =int)
    parser.add_argument("--criterion", default='entropy', help ="criterion for base estimator")
    parser.add_argument("--max_depth", default=10, help = "maximum depth for base estimator", type = int)
    parser.add_argument("--epsilon", default =0.5, help = "epsilon value for optimization", type = int)
    parser.add_argument("--no_adv_images", default=50, help = "number of adversarial to be generated for optimization", type = int)

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
    global base_estimator_params

    heat_map_path = args_dict['heat_map_path']
    base_estimator = args_dict['base_estimator']
    n_estimators = args_dict['n_estimators']
    criterion = args_dict['criterion']
    max_depth = args_dict['max_depth']
    epsilon = args_dict['epsilon']
    no_adv_images = args_dict['no_adv_images']

    
    if(base_estimator == "random_forest"):
    	base_estimator = RandomForestClassifier()

    base_classifier_params = {'n_estimators' : n_estimators,\
						  'criterion' : criterion,\
						  'max_depth' : max_depth}


	inner_iter = [5, 10, 20, 50, 100]
	outer_iter = [1, 2, 5, 10]

	acc_normal_list = []
	acc_adv_images_list = []
	misclassification_list = []

	#Offline running to calculate accuracy
	for in_iter in inner_iter:
		acc_normal_out = []
		acc_adv_images_out = []
		misclassification_out = []
		for out_iter in outer_iter:

			#Read image_vecs, noise_vecs, and correct_labels
			read_image_noise_labels_from_file()
			#Calculate accuracy and display
			acc_normal, acc_adv_images, misclassification = calc_accuracy_and_display(n_estimators, epsilon, clf, in_iter, out_iter, \
																						no_adv_images, image_vecs, noise_vecs, correct_labels) #Can be done offline
			
			acc_normal_out.append(acc_normal)
			acc_adv_images_out.append(acc_adv_images)
			misclassification_out.append(misclassification)

		acc_normal_list.append(acc_normal_out)
		acc_adv_images_list.append(acc_adv_images_out)
		misclassification_list.append(misclassification_out)

	#Plot heatmap

	plot_heatmap(acc_normal_list, acc_adv_list, misclassification_list)

if __name__ == '__main__':

	
	main()