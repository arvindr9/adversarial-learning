from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from scipy.optimize import minimize
from scipy.linalg import norm
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.optimize import basinhopping
import csv
from sklearn.metrics import accuracy_score


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
rand_idx = np.random.randint(0,len(eval_data), (500,))
eval_data_ss = eval_data[rand_idx]
eval_labels_ss = eval_labels[rand_idx]

def fitness_func(noise, x, clf):

	x = np.array(x).reshape(1,-1)
	return -1*norm(clf.predict_proba(x) - clf.predict_proba(x+noise))

n_est = [1, 2, 4, 5, 10, 50, 100]
accuracy = []
for est in n_est:

	# Training Random forest classifier and predicting
	print('Training Random Forest with no of estimators: {}'.format(est))

	clf = RandomForestClassifier(n_estimators = est, criterion = 'entropy', max_depth =10)
	clf.fit(train_data_ss, train_labels_ss)
	accuracy.append(accuracy_score(eval_labels_ss, clf.predict(eval_data_ss)))



	bounds = [(-0.5,0.5)]
	x0 = [0]
	iterations = [100, 200, 500, 1000]


	mean_fitness = []
	mean_noise = []
	var_noise = []
	min_noise = []
	max_noise = []
	wrong_output= []

	print('Starting the optimization')
	for i in iterations:

		optimal_fitness = []
		optimal_noise = []
		noise_label = []
		correct_label = []

		print('Current Iteration: {}'.format(i))
		for image_no, image in enumerate(eval_data_ss[1:100,:]):

			print('Current Image: {}'.format(image_no))
			minimizer_kwargs = dict(method = "L-BFGS-B", bounds = bounds, args = (image,clf))
			res = basinhopping(fitness_func, niter = i, x0 = x0, minimizer_kwargs = minimizer_kwargs)
			optimal_fitness.append(res['fun'])
			optimal_noise.append(res['x'])
			correct_label.append(clf.predict(image.reshape(1,-1))[0])
			noise_label.append(clf.predict((image+res['x'][0]).reshape(1,-1))[0])

		mean_fitness.append(np.mean(optimal_fitness)*(-1))
		mean_noise.append(np.mean(optimal_noise))
		min_noise.append(np.min(optimal_noise))
		max_noise.append(np.max(optimal_noise))
		var_noise.append(np.std(optimal_noise))

		correct_label = np.array(correct_label)
		noise_label = np.array(noise_label)

		wrong_output.append((len(correct_label[correct_label!=noise_label])/len(correct_label))*100)

	mean_fitness = np.array(mean_fitness).reshape(-1,1)
	mean_noise = np.array(mean_noise).reshape(-1,1)
	var_noise = np.array(var_noise).reshape(-1,1)
	min_noise = np.array(min_noise).reshape(-1,1)
	max_noise = np.array(max_noise).reshape(-1,1)
	wrong_output = np.array(wrong_output).reshape(-1,1)

	data = np.concatenate((mean_fitness, mean_noise, var_noise, min_noise, max_noise, wrong_output), axis = 1)


	#Writing data to file
	file = './data' + str(est) + '.csv'
	with open(file, 'w', newline = '') as output:
		writer = csv.writer(output, delimiter=',')
		writer.writerows(data)

	#Results
	f1, ax1 = plt.subplots()
	ax1.plot(iterations, mean_fitness)
	plt.title('Mean Optimal fitness vs iterations')
	plt.xlabel('Iterations')
	plt.ylabel('Mean Optimal fitness values')
	plt_name=  'plot_fitness'+str(est) +'.png'
	plt.savefig(plt_name)

	f2, ax2 = plt.subplots()
	ax2.plot(iterations, mean_noise)
	plt.title('Mean Optimal noise vs iterations')
	plt.xlabel('Iterations')
	plt.ylabel('Mean Optimal noise')
	plt_name = 'plot_noise' + str(est) + '.png'
	plt.savefig(plt_name)


	f3, ax3 = plt.subplots()
	ax3.plot(iterations, var_noise)
	plt.title('Noise std vs iterations')
	plt.xlabel('Iterations')
	plt.ylabel('Std deviation')
	plt_name = 'plot_variance' + str(est) + '.png'
	plt.savefig(plt_name)

	f4, ax4 = plt.subplots()
	ax4.plot(iterations, min_noise)
	plt.title('Min noise vs iterations')
	plt.xlabel('Iterations')
	plt.ylabel('Min Noise')
	plt_name = 'plot_min' + str(est) + '.png'
	plt.savefig(plt_name)

	f5, ax5 = plt.subplots()
	ax5.plot(iterations, max_noise)
	plt.title('Max noise vs iterations')
	plt.xlabel('Iterations')
	plt.ylabel('Max Noise')
	plt_name = 'plot_max' + str(est) + '.png'
	plt.savefig(plt_name)

	f6, ax6 = plt.subplots()
	ax6.plot(iterations, wrong_output)
	plt.title('Percentage misclassified vs iterations')
	plt.xlabel('Iterations')
	plt.ylabel('Percentage misclassified')
	plt_name = 'plot_misclassified' + str(est) + '.png'
	plt.savefig(plt_name)

accuracy = np.array(accuracy).reshape(-1,1)
file = './accuracy.csv'
with open(file, 'w', newline = '') as output:
	writer = csv.writer(output, delimiter=',')
	writer.writerows(accuracy)

print(accuracy)
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

	



