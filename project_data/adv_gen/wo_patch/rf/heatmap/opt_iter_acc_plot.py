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
# import cPickle
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
	acc_adv_images = accuracy_score(np.array(correct_labels[0]), adv_image_label)

	misclassification = (len(normal_image_label[normal_image_label!=adv_image_label])/len(normal_image_label))*100

	return acc_normal, acc_adv_images, misclassification


def read_image_noise_labels_from_file(in_iter, out_iter):

	#image_vecs
	file = heat_map_path + '/image_data' + '/' + list(base_classifier_params.keys())[0] + '_' + str(list(base_classifier_params.values())[0]) +\
			 '_' + 'eps_' + str(epsilon) + '_'+ 'in_'+ str(in_iter) + '_out_'+ str(out_iter) +'.csv'
	with open(file, 'r') as filehandle:
		filecontents = filehandle.readlines()
	image_vecs = []
	for image in filecontents:
		image_vecs.append([float(x) for x in image.split(',')])


	#noise_vecs
	file = heat_map_path + '/noise' +'/'+ list(base_classifier_params.keys())[0]  + '_' + str(list(base_classifier_params.values())[0]) +\
			 '_' + 'eps_' + str(epsilon) + '_'+ 'in_'+ str(in_iter) + '_out_'+ str(out_iter) +'.csv'
	with open(file, 'r') as filehandle:
		filecontents = filehandle.readlines()
	noise_vecs = []
	for noise in filecontents:
		noise_vecs.append([float(x) for x in noise.split(',')])


	#correct_labels
	file = heat_map_path + '/true_labels' +'/'+ list(base_classifier_params.keys())[0]  + '_' + str(list(base_classifier_params.values())[0]) +\
		  '_' + 'eps_' + str(epsilon) + '_'+ 'in_'+ str(in_iter) + '_out_'+ str(out_iter) +'.csv'
	with open(file, 'r') as filehandle:
		filecontents = filehandle.readlines()
	correct_labels = []
	for label in filecontents:
		correct_labels.append([float(x) for x in label.split(',')])

	return np.array(image_vecs), np.array(noise_vecs), np.array(correct_labels)


def plot_and_save_heatmap(acc_normal_list, acc_adv_list, misclassification_list, inner_iter, outer_iter):

	try:
		import seaborn as sns
		import pandas as pd
		import matplotlib.pyplot as plt

	except:
		raise ValueError('Libraries : seaborn and pandas not installed')

	sns.set()


	cols = pd.Index([str(x) for x in outer_iter], name = 'Outer Iterations')
	rows = pd.Index([str(x) for x in inner_iter], name = 'Inner Iterations')
	acc_normal_list_df = pd.DataFrame(acc_normal_list, index = rows, columns = cols)

	acc_adv_list_df = pd.DataFrame(acc_adv_list, index = rows, columns= cols)
	print(acc_adv_list_df)
	misclassification_list_df = pd.DataFrame(misclassification_list, index =rows, columns = cols)
	heatmap_normal = sns.heatmap(acc_normal_list_df, annot=True)
	plt.title('Accuracy on normal images')
	fig = heatmap_normal.get_figure()
	fig.savefig('heatmap_normal.pdf')
	plt.close()
	heatmap_adv = sns.heatmap(acc_adv_list_df, annot= True)
	plt.title('Accuracy on adversarial images')
	fig = heatmap_adv.get_figure()
	fig.savefig('heatmap_adversarial.pdf')
	plt.close()

	heatmap_mis = sns.heatmap(misclassification_list_df, annot= True)
	plt.title('Misclassification rate')
	fig = heatmap_mis.get_figure()
	fig.savefig('heatmap_mis.pdf')
	


def main():

	parser = argparse.ArgumentParser()
	parser.add_argument("--heat_map_path", default = "/home/siddharth/Desktop/Adversarial Learning SP/new_folder/adversarial-learning/project_data/adv_gen/wo_patch/rf/heatmap/heatmap_data", help="where all the data and files related to heat map gen are stored")
	parser.add_argument("--base_estimator", default = "random_forest", help="base estimator {'random_forest'}")
	parser.add_argument("--n_estimators", default = 20, help ="no. of estimators in base estimators", type =int)
	parser.add_argument("--criterion", default = 'entropy', help ="criterion for base estimator")
	parser.add_argument("--max_depth", default = 10, help = "maximum depth for base estimator", type = int)
	parser.add_argument("--epsilon", default = 0.5, help = "epsilon value for optimization", type = int)
	parser.add_argument("--no_adv_images", default = 50, help = "number of adversarial to be generated for optimization", type = int)
	parser.add_argument("--no_of_threads", default = 70, help = "number of threads to run in parallel", type = int)
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

	clf = joblib.load('./heat_map_trained_classifier.pkl')
	
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
			image_vecs, noise_vecs, correct_labels = read_image_noise_labels_from_file(in_iter, out_iter)
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

	plot_and_save_heatmap(acc_normal_list, acc_adv_images_list, misclassification_list, inner_iter, outer_iter)

if __name__ == '__main__':

	
	main()