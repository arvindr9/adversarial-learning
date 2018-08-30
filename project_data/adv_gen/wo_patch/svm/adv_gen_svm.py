from sklearn import tree
from sklearn.svm import SVC
from scipy.optimize import minimize
from scipy.linalg import norm
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.optimize import basinhopping
import csv
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
import cPickle

'''
TODO:
Increase number of testing samples for finding accuracy (let's say 1000-5000)
Parallelize for the C and eps values
'''

def save_object(obj, filename):
   with open(filename, 'wb') as output:  # Overwrites any existing file.
       cPickle.dump(obj, output, cPickle.HIGHEST_PROTOCOL)

#Loading Mnist data
# Load training and eval data
mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_data = mnist.train.images  # Returns np.array
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
eval_data = mnist.test.images  # Returns np.array
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)


#Subsampling the training dataset
rand_idx = np.random.randint(0,len(train_data), (10000,))
train_data_ss = train_data[rand_idx]
train_labels_ss = train_labels[rand_idx]

#Writing permutation of train data to a file
file = 'raw_data/permutations/perm_train.csv'
with open(file, 'w') as output:
	writer = csv.writer(output, delimiter = ',')
	writer.writerow(rand_idx)

#Subsampling the testing dataset
rand_idx = np.random.randint(0,len(eval_data), (5000,))
eval_data_ss = eval_data[rand_idx]
eval_labels_ss = eval_labels[rand_idx]

#Writing permutation of test data to a file
file = 'raw_data/permutations/perm_test.csv'
with open(file, 'w') as output:
	writer = csv.writer(output, delimiter = ',')
	writer.writerow(rand_idx)

def fitness_func(noise, x, clf):

	x = np.array(x).reshape(1,-1)
	return -1*norm(clf.predict_proba(x) - clf.predict_proba(x+noise))

accuracy = []
Cs = [0.01, 0.1, 1.0, 10, 100]
epsilons = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

#Save classifiers
clfs = {}
for c in Cs:
	clfs[str(c)] = SVC(C = c, probability=True)
	clf = clfs[str(c)]
	clf.fit(train_data_ss, train_labels_ss)
	accuracy.append(accuracy_score(eval_labels_ss, clf.predict(eval_data_ss)))
	print("score: {}".format(accuracy_score(eval_labels_ss, clf.predict(eval_data_ss))))
save_object(clfs, 'raw_data/clfs.pkl')

def advGen(c, epsilon):

	# Training Random forest classifier and predicting
	# print('Training Random Forest with no of estimators: {}'.format(est))

	# clf = RandomForestClassifier(n_estimators = est, criterion = 'entropy', max_depth =10)
	# print('Created classifier')
	# clf.fit(train_data_ss, train_labels_ss)
	# print('Trained classifier')
	# accuracy.append(accuracy_score(eval_labels_ss, clf.predict(eval_data_ss)))

	# print(f"score: {accuracy_score(eval_labels_ss, clf.predict(eval_data_ss))}")
	clf = clfs[str(c)]

	x0 = [0]*784
	#iterations = [100, 200, 500, 1000]
	#epsilons = [0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 1.0]


	'''
	CSV File contains:
	0. mean_fitness
	1. mean_noise
	2. var_noise, min_noise, max_noise, wrong_output, mean_l0,
	min_l0, max_l0, var_l0, mean_l1, min_l1, max_l1, var_l1
	'''

	print("noise norm: {}".format(norm(x0)))

	print('Starting the optimization')

	# optimal_fitness = []
	# optimal_noise = []
	# noise_label = []
	correct_labels = []
	predictions = []
	# optimal_l0 = []
	# optimal_l1 = []
	noise_vecs = []
	image_vecs = []


	print('Current C: {}, Current Epsilon: {}'.format(c, epsilon))
	for image_no, image in enumerate(eval_data_ss[1:501,:]):
		x0 = [0] * 784
		print('Current Image: {}'.format(image_no))
		cons = ({'type': 'ineq',
			'fun': lambda noise: epsilon - norm(np.array(noise))})
		minimizer_kwargs = dict(method = "slsqp", args = (image,clf), constraints = cons)
		res = basinhopping(fitness_func, niter = 10, x0 = x0, minimizer_kwargs = minimizer_kwargs)
		image_vecs.append(image)
		noise_vecs.append(res['x'])
		correct_labels.append(eval_labels_ss[image_no + 1])
		predictions.append(clf.predict(image + np.array(res['x']).reshape(1, -1)))

	#Writing data to file
	file = 'raw_data/images/c_' + str(c) + 'eps_' + str(epsilon) + '.csv'
	with open(file, 'w') as output:
		writer = csv.writer(output, delimiter=',')
		writer.writerows(image_vecs)
	file = 'raw_data/noise/c_' + str(c) + 'eps_' + str(epsilon) + '.csv'
	with open(file, 'w') as output:
		writer = csv.writer(output, delimiter = ',')
		writer.writerows(noise_vecs)
	print(correct_labels)
	file = 'raw_data/correct_labels/c_' + str(c) + 'eps_' + str(epsilon) + '.csv'
	with open(file, 'w') as output:
		writer = csv.writer(output, delimiter = ',')
		writer.writerow(correct_labels)
	file = 'raw_data/predictions/c_' + str(c) + 'eps_' + str(epsilon) + '.csv'
	with open(file, 'w') as output:
		writer = csv.writer(output, delimiter = ',')
		writer.writerow(predictions)


accuracy = np.array(accuracy).reshape(-1,1)
print(accuracy)
file = 'raw_data/accuracy.csv'
with open(file, 'w') as output:
	writer = csv.writer(output, delimiter=',')
	writer.writerows(accuracy)

if __name__ == '__main__':
	tasks = []
	for c in Cs:
		for eps in epsilons:
			tasks.append((c, eps))
	Parallel(n_jobs=70)(delayed(advGen)(c, eps) for (c, eps) in tasks)


