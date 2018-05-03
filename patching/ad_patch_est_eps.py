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
import random
from joblib import Parallel, delayed

patch_size = 3
n_adv_images = 101

#Generating random start row and start col points for the patch for each image for adversarial training
# random_start_row = []
# random_start_col = []

# for i in range(n_adv_images):

# 	random_start_row.append(random.randint(1,28-patch_size))
# 	random_start_col.append(random.randint(1,28-patch_size))


def add_patch_to_image(image, patch, patch_size, patch_start_row, patch_start_col):

	"""
	Adding the patch noise to the image

	Input:
	------
	image: flattened out image
	patch: flattened out patch

	Returns:
	--------

	flat_image: flattened out image with patch added

	"""

	reshape_image = np.array(image).reshape(28,28)
	reshape_patch = np.array(patch).reshape(patch_size, patch_size)

	reshape_image[patch_start_row : patch_start_row + patch_size, patch_start_col : patch_start_col + patch_size] += reshape_patch

	return reshape_image.reshape(1,28*28)


def fitness_func(noise, x, clf, image_no, random_start_row, random_start_col, patch_size):

	x = np.array(x).reshape(1,-1)
	patch_image = add_patch_to_image(x, noise, patch_size, random_start_row, random_start_col)
	return -1*norm(clf.predict_proba(x) - clf.predict_proba(patch_image))


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


n_est = [1, 2, 5, 10, 20, 50, 100]
accuracy = []

clfs = {}
for est in n_est:
	clfs[str(est)] = RandomForestClassifier(n_estimators = est, criterion = 'entropy', max_depth =10)
	clf = clfs[str(est)]
	clf.fit(train_data_ss, train_labels_ss)
	accuracy.append(accuracy_score(eval_labels_ss, clf.predict(eval_data_ss)))
	print("score: {}".format(accuracy_score(eval_labels_ss, clf.predict(eval_data_ss))))

epsilons = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]

def advGen(est, epsilon):

    clf = clfs[str(est)]

    x0 = [0]*patch_size*patch_size
    # iterations = [100, 200, 500, 1000]
    # epsilons = [0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 1.0]


    '''
    TODO:
    Plot:
    Single plot for data from multiple estimators
    CSV File contains:
    0. mean_fitness
    1. mean_noise
    2. var_noise, min_noise, max_noise, wrong_output, mean_l0,
    min_l0, max_l0, var_l0, mean_l1, min_l1, max_l1, var_l1
    '''

    print('Starting the optimization')

    optimal_fitness = []
    optimal_noise = []
    noise_label = []
    correct_label = []
    optimal_l0 = []
    optimal_l1 = []


    print('Current Epsilon: {}'.format(epsilon))
    for image_no, image in enumerate(eval_data_ss[1:n_adv_images,:]):
        x0 = [0] * patch_size * patch_size
        print('Current Image: {}'.format(image_no))
        cons = ({'type': 'ineq','fun': lambda noise: epsilon - norm(np.array(noise))})
        min_fitness = 0
        res = None
        row_opt = None
        col_opt = None
        for row in range(0, 25):
            for col in range(0, 25):
                minimizer_kwargs = dict(method = "slsqp", args = (image, clf, image_no, row, col, patch_size), constraints = cons)
                curr_res = basinhopping(fitness_func, niter = 1, x0 = x0, minimizer_kwargs = minimizer_kwargs)
                curr_fitness = curr_res['fun']
                if curr_fitness <= min_fitness:
                    min_fitness = curr_fitness
                    res = curr_res
                    row_opt = row
                    col_opt = col
        optimal_fitness.append(res['fun'])
        noise_norm = norm(res['x'])
        l0_norm = norm(res['x'], ord = 0)
        l1_norm = norm(res['x'], ord = 1)

        optimal_noise.append(noise_norm)
        optimal_l0.append(l0_norm)
        optimal_l1.append(l1_norm)

        correct_label.append(clf.predict(image.reshape(1,-1))[0])
        patch_image = add_patch_to_image(image, res['x'], patch_size, row_opt, col_opt)
        noise_label.append(clf.predict((patch_image).reshape(1,-1))[0])


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
            plt.imshow(patch_image.reshape([28, 28]), cmap=plt.get_cmap('gray_r'))
            fig.add_subplot(1, 3, 3)
            plt.imshow(np.array(res['x']).reshape([3, 3]), cmap=plt.get_cmap('gray_r'))
            plt_name = 'est_eps_patch/est' + str(est) + '_epsilon' + str(epsilon) + '_image' + str(image_no) + '.png'
            plt.savefig(plt_name)


    data = np.concatenate((optimal_fitness, optimal_noise, noise_label, correct_label, optimal_l0, optimal_l1))

	#Writing data to file
    file = 'est_eps_patch/est_' + str(est) + 'eps' + str(epsilon) + '.csv'
    with open(file, 'wb') as output:
        writer = csv.writer(output, delimiter=',')
        writer.writerow(data)


accuracy = np.array(accuracy).reshape(-1,1)
file = 'est_eps_patch/accuracy.csv'
with open(file, 'wb') as output:
	writer = csv.writer(output, delimiter=',')
	writer.writerows(accuracy)

print(accuracy)

if __name__ == '__main__':
    tasks = []
    for est in n_est:
        for eps in epsilons:
            tasks.append((est, eps))

    Parallel(n_jobs=63)(delayed(advGen)(est, epsilon) for (est, epsilon) in tasks)