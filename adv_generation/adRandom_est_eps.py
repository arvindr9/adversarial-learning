from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
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

'''
TODO:
Increase number of testing samples for finding accuracy (let's say 1000-5000)
Parallelize for the estimator and eps values
'''


#Loading Mnist data
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

def fitness_func(noise, x, clf):

	x = np.array(x).reshape(1,-1)
	return -1*norm(clf.predict_proba(x) - clf.predict_proba(x+noise))

n_est = [1, 2, 5, 10, 20, 50, 100]
accuracy = []
epsilons = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0]

'''
Want to train each classifier only once
pretrain classifiers
results = {}
results['2_est'].mean_fitness, etc...
'''
clfs = {}
for est in n_est:
	clfs[str(est)] = RandomForestClassifier(n_estimators = est, criterion = 'entropy', max_depth =10)
	clf = clfs[str(est)]
	clf.fit(train_data_ss, train_labels_ss)
	accuracy.append(accuracy_score(eval_labels_ss, clf.predict(eval_data_ss)))
	print("score: {}".format(accuracy_score(eval_labels_ss, clf.predict(eval_data_ss))))

def advGen(est, epsilon):

	# Training Random forest classifier and predicting
	# print('Training Random Forest with no of estimators: {}'.format(est))

	# clf = RandomForestClassifier(n_estimators = est, criterion = 'entropy', max_depth =10)
	# print('Created classifier')
	# clf.fit(train_data_ss, train_labels_ss)
	# print('Trained classifier')
	# accuracy.append(accuracy_score(eval_labels_ss, clf.predict(eval_data_ss)))

	# print(f"score: {accuracy_score(eval_labels_ss, clf.predict(eval_data_ss))}")
	clf = clfs[str(est)]

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

	optimal_fitness = []
	optimal_noise = []
	noise_label = []
	correct_label = []
	optimal_l0 = []
	optimal_l1 = []


	print('Current Estimator: {}, Current Epsilon: {}'.format(est, epsilon))
	for image_no, image in enumerate(eval_data_ss[1:n_adv_images,:]):
		x0 = [0] * 784
		print('Current Image: {}'.format(image_no))
		cons = ({'type': 'ineq',
			'fun': lambda noise: epsilon - norm(np.array(noise))})
		minimizer_kwargs = dict(method = "slsqp", args = (image,clf), constraints = cons)
		res = basinhopping(fitness_func, niter = 10, x0 = x0, minimizer_kwargs = minimizer_kwargs)
		optimal_fitness.append(res['fun'])
		noise_norm = norm(res['x'])
		l0_norm = norm(res['x'], ord = 0)
		l1_norm = norm(res['x'], ord = 1)

		optimal_noise.append(noise_norm)
		optimal_l0.append(l0_norm)
		optimal_l1.append(l1_norm)
		#print("noise norm: {}".format(noise_norm))
		correct_label.append(clf.predict(image.reshape(1,-1))[0])
		noise_label.append(clf.predict((image+res['x']).reshape(1,-1))[0])
		if correct_label[-1] != noise_label[-1]:
		    fig = plt.figure()
		    rows = 1
		    columns = 3
		    fig.text(0.15, 0.05, "original label: {}".format(correct_label[-1]))
		    fig.text(0.4, 0.05, "adversarial label: {}".format(noise_label[-1]))
		    fig.text(0.275, 0.15, "real class: {}".format(eval_labels_ss[1:n_adv_images][image_no]))
		    fig.add_subplot(1, 3, 1)
		    plt.imshow(image.reshape([28,28]), cmap=plt.get_cmap('gray_r'))
		    fig.add_subplot(1, 3, 2)
		    plt.imshow((image + res['x']).reshape([28, 28]), cmap=plt.get_cmap('gray_r'))
		    fig.add_subplot(1, 3, 3)
		    plt.imshow(np.array(res['x']).reshape([28, 28]), cmap=plt.get_cmap('gray_r'))
		    plt_name = 'est_eps/est' + str(est) + '_epsilon' + str(epsilon) + '_image' + str(image_no) + '.png'
		    plt.savefig(plt_name)


	data = np.concatenate((optimal_fitness, optimal_noise, noise_label, correct_label, optimal_l0, optimal_l1))

	#Writing data to file
	file = 'est_eps/est_' + str(est) + 'eps' + str(epsilon) + '.csv'
	with open(file, 'wb') as output:
		writer = csv.writer(output, delimiter=',')
		writer.writerow(data)






accuracy = np.array(accuracy).reshape(-1,1)
file = 'est_eps/accuracy.csv'
with open(file, 'wb') as output:
	writer = csv.writer(output, delimiter=',')
	writer.writerows(accuracy)

print(accuracy)

if __name__ == '__main__':
	print('hi')
	tasks = []
	for est in n_est:
		for eps in epsilons:
			tasks.append((est, eps))
	Parallel(n_jobs=42)(delayed(advGen)(est, eps) for (est, eps) in tasks)




# plt.subplot(121)
# plt.imshow(image.reshape(28,28))
# plt.title('Classified label without noise = %d'%(clf.predict(image.reshape(1,-1))))

# plt.subplot(122)
# plt.imshow((image+res['x'][0]).reshape(28,28))
# plt.title('Classified label with optimal noise = %d'%(clf.predict((image+res['x'][0]).reshape(1,-1))))

# plt.show()

# # print(clf.tree_.predict(eval_data_ss))
# # #Calculating error
# # error = (len(eval_labels_ss) -  len(eval_labels_ss[eval_labels_ss == y_pred]))/len(eval_labels_ss)
# # print(error)

# def distance_func(x_noisy, x):
# 	"""
# 	Returns the 2nd norm distance between noisy image and original image
# 	"""
# 	return np.square(norm((x_noisy - x), ord = 2))

# def true_class(i, type):

# 	if(type == 'train'):

# 		return train_labels[i]

# 	elif(type == 'eval'):

# 		return train_labels[i]

# def classified_class(x):

# 	return clf.predict(x)

# def add_noise(x):

	


