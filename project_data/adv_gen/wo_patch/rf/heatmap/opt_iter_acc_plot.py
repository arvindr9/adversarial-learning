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


#Offline running to calculate accuracy
for in_iter in inner_iter:
	for out_iter in outer_iter:

		#Read image_vecs, noise_vecs, and correct_labels
		read_image_noise_labels_from_file()
		#Calculate accuracy and display
		acc_normal, acc_adv_images, misclassification = calc_accuracy_and_display(n_estimators, epsilon, clf, in_iter, out_iter, no_adv_images, image_vecs, noise_vecs, correct_labels) #Can be done offline
		
		#Plot heatmap
		plot_heatmap()