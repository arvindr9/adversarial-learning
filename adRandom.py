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

	print(f"score: {accuracy_score(eval_labels_ss, clf.predict(eval_data_ss))}")

	x0 = [0]*784
	#iterations = [100, 200, 500, 1000]
	epsilons = [0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 1.0]

	'''
	TODO:
	Plot:
	Epsilon vs accuracy
	Confidence bands
	Single plot for data from multiple estimators
	'''

	mean_fitness = []
	mean_noise = []
	var_noise = []
	min_noise = []
	max_noise = []
	wrong_output= []
	mean_l0 = []
	min_l0 = []
	var_l0 = []
	max_l0 = []
	mean_l1 = []
	min_l1 = []
	max_l1 = []
	var_l1 = []
	print(f"noise norm: {norm(x0)}")

	print('Starting the optimization')
	for epsilon in epsilons:

		optimal_fitness = []
		optimal_noise = []
		noise_label = []
		correct_label = []
		optimal_l0 = []
		optimal_l1 = []


		print('Current Epsilon: {}'.format(epsilon))
		for image_no, image in enumerate(eval_data_ss[1:100,:]):
			x0 = [0] * 784
			print('Current Image: {}'.format(image_no))
			cons = ({'type': 'ineq',
					'fun': lambda noise: epsilon - norm(np.array(noise))})
			minimizer_kwargs = dict(method = "slsqp", args = (image,clf), constraints = cons, options = {'maxiter': 100})
			res = basinhopping(fitness_func, niter = 1, x0 = x0, minimizer_kwargs = minimizer_kwargs)
			optimal_fitness.append(res['fun'])
			noise_norm = norm(res['x'])
			l0_norm = norm(res['x'], ord = 0)
			l1_norm = norm(res['x'], ord = 1)
			
			optimal_noise.append(noise_norm)
			optimal_l0.append(l0_norm)
			optimal_l1.append(l1_norm)
			#print(f"noise norm: {noise_norm}")
			correct_label.append(clf.predict(image.reshape(1,-1))[0])
			noise_label.append(clf.predict((image+res['x'][0]).reshape(1,-1))[0])

		mean_fitness.append(np.mean(optimal_fitness)*(-1))
		mean_noise.append(np.mean(optimal_noise))
		min_noise.append(np.min(optimal_noise))
		max_noise.append(np.max(optimal_noise))
		var_noise.append(np.std(optimal_noise))

		mean_l0.append(np.mean(optimal_l0))
		min_l0.append(np.min(optimal_l0))
		max_l0.append(np.max(optimal_l0))
		var_noise.append(np.std(optimal_l0))


		mean_l1.append(np.mean(optimal_l1))
		min_l1.append(np.min(optimal_l1))
		max_l1.append(np.max(optimal_l1))
		var_noise.append(np.std(optimal_l1))

		correct_label = np.array(correct_label)
		noise_label = np.array(noise_label)

		wrong_output.append((len(correct_label[correct_label!=noise_label])/len(correct_label))*100)

	mean_fitness = np.array(mean_fitness).reshape(-1,1)
	mean_noise = np.array(mean_noise).reshape(-1,1)
	var_noise = np.array(var_noise).reshape(-1,1)
	min_noise = np.array(min_noise).reshape(-1,1)
	max_noise = np.array(max_noise).reshape(-1,1)
	mean_l0 = np.array(mean_l0).reshape(-1, 1)
	min_l0 = np.array(min_l0).reshape(1, -1)
	max_l0 = np.array(max_l0).reshape(1, -1)
	var_l0 = np.array(var_l0).reshape(1, -1)
	mean_l1 = np.array(mean_l1).reshape(-1, 1)
	min_l1 = np.array(min_l0).reshape(-1, 1)
	max_l1 = np.array(max_l1).reshape(-1, 1)
	wrong_output = np.array(wrong_output).reshape(-1,1)

	data = np.concatenate((mean_fitness, mean_noise, var_noise, min_noise, max_noise, wrong_output, mean_l0, min_l0, max_l0, var_l0, mean_l1, min_l1, max_l1, var_l1), axis = 1)
	


	#Writing data to file
	file = './data' + str(est) + '.csv'
	with open(file, 'w', newline = '') as output:
		writer = csv.writer(output, delimiter=',')
		writer.writerows(data)

	#Results
	f1, ax1 = plt.subplots()
	ax1.plot(epsilons, mean_fitness)
	plt.title('Mean Optimal fitness vs epsilon')
	plt.xlabel('Epsilon')
	plt.ylabel('Mean Optimal fitness values')
	plt_name=  'plot_fitness'+str(est) +'.png'
	plt.savefig(plt_name)

	f2, ax2 = plt.subplots()
	ax2.plot(epsilons, mean_noise)
	plt.title('Mean Optimal noise vs epsilon')
	plt.xlabel('Epsilon')
	plt.ylabel('Mean Optimal noise')
	plt_name = 'plot_noise' + str(est) + '.png'
	plt.savefig(plt_name)

	f3, ax3 = plt.subplots()
	ax3.plot(epsilons, mean_noise)
	ax3.fill_between(epsilons, list(np.ndarray.flatten(np.array(mean_noise) + np.array(var_noise))), list(np.ndarray.flatten(np.array(mean_noise) - np.array(var_noise))))
	plt.title('Optimal noise distribution vs epsilon')
	plt.xlabel('Epsilon')
	plt.ylabel('Optimal Noise Distribution')
	plt_name = 'plot_distribution' + str(est) + '.png'
	plt.savefig(plt_name)

	#eventually change the above to be a graph that represents using all possible numbers of estimators.
	
	# f3, ax3 = plt.subplots()
	# ax3.plot(epsilons, var_noise)
	# plt.title('Noise std vs epsilon')
	# plt.xlabel('Epsilon')
	# plt.ylabel('Std deviation')
	# plt_name = 'plot_variance' + str(est) + '.png'
	# plt.savefig(plt_name)

	f4, ax4 = plt.subplots()
	ax4.plot(epsilons, min_noise)
	plt.title('Min noise vs epsilon')
	plt.xlabel('Epsilon')
	plt.ylabel('Min Noise')
	plt_name = 'plot_min' + str(est) + '.png'
	plt.savefig(plt_name)

	f5, ax5 = plt.subplots()
	ax5.plot(epsilons, max_noise)
	plt.title('Max noise vs epsilon')
	plt.xlabel('Epsilon')
	plt.ylabel('Max Noise')
	plt_name = 'plot_max' + str(est) + '.png'
	plt.savefig(plt_name)

	f6, ax6 = plt.subplots()
	ax6.plot(epsilons, wrong_output)
	plt.title('Percentage misclassified vs epsilon')
	plt.xlabel('Epsilon')
	plt.ylabel('Percentage misclassified')
	plt_name = 'plot_misclassified' + str(est) + '.png'
	plt.savefig(plt_name)

	#L_0 norm
	f7, ax7 = plt.subplots()
	ax7.plot(epsilons, mean_l0)
	plt.title('Mean Optimal L0 norm vs epsilon')
	plt.xlabel('Epsilon')
	plt.ylabel('Mean L0 norm')
	plt_name = 'l0_' + str(est) + '.png'
	plt.savefig(plt_name)

	f8, ax8 = plt.subplots()
	ax8.plot(epsilons, min_l0)
	plt.title('Min L0 norm vs epsilon')
	plt.xlabel('Epsilon')
	plt.ylabel('Min L0 norm')
	plt_name = 'plot_min_l0_' + str(est) + '.png'
	plt.savefig(plt_name)

	f9, ax9 = plt.subplots()
	ax9.plot(epsilons, max_l0)
	plt.title('Max L0 norm vs epsilon')
	plt.xlabel('Epsilon')
	plt.ylabel('Max L0 norm')
	plt_name = 'plot_max_l0_' + str(est) + '.png'
	plt.savefig(plt_name)

	f10, ax10 = plt.subplots()
	ax10.plot(epsilons, mean_l0, 'k-')
	ax10.fill_between(epsilons, list(np.ndarray.flatten(np.array(mean_l0) + np.array(var_l0))), list(np.ndarray.flatten(np.array(mean_l0) - np.array(var_l0))))
	plt.title('L0 norm distribution vs epsilon')
	plt.xlabel('Epsilon')
	plt.ylabel('L0 norm')
	plt_name = 'plot_distribution_l0_' + str(est) + '.png'
	plt.savefig(plt_name)

	#L_1 norm
	f11, ax11 = plt.subplots()
	ax11.plot(epsilons, mean_l1)
	plt.title('Mean Optimal L1 norm vs epsilon')
	plt.xlabel('Epsilon')
	plt.ylabel('Mean L1 norm')
	plt_name = 'l1_' + str(est) + '.png'
	plt.savefig(plt_name)

	f12, ax12 = plt.subplots()
	ax8.plot(epsilons, min_l1, 'k-')
	plt.title('Min L1 norm vs epsilon')
	plt.xlabel('Epsilon')
	plt.ylabel('Min L1 norm')
	plt_name = 'plot_min_l1_' + str(est) + '.png'
	plt.savefig(plt_name)

	f13, ax13 = plt.subplots()
	ax13.plot(epsilons, max_l1)
	plt.title('Max L1 norm vs epsilon')
	plt.xlabel('Epsilon')
	plt.ylabel('Max L1 norm')
	plt_name = 'plot_max_l1_' + str(est) + '.png'
	plt.savefig(plt_name)

	f14, ax14 = plt.subplots()
	ax14.plot(epsilons, mean_l1)
	ax14.fill_between(epsilons, list(np.ndarray.flatten(np.array(mean_l1) + np.array(var_l1))), list(np.ndarray.flatten(np.array(mean_l1) - np.array(var_l1))))
	plt.title('L1 norm distribution vs epsilon')
	plt.xlabel('Epsilon')
	plt.ylabel('L1 norm')
	plt_name = 'plot_distribution_l1_' + str(est) + '.png'
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

	



