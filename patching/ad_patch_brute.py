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

patch_size = 3
n_adv_images = 100

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
rand_idx = np.random.randint(0,len(eval_data), (500,))
eval_data_ss = eval_data[rand_idx]
eval_labels_ss = eval_labels[rand_idx]


n_est = [1, 5, 10, 50, 100]
accuracy = []
for est in n_est:

    # Training Random forest classifier and predicting
    print('Training Random Forest with no of estimators: {}'.format(est))

    clf = RandomForestClassifier(n_estimators = est, criterion = 'entropy', max_depth =10)
    clf.fit(train_data_ss, train_labels_ss)
    accuracy.append(accuracy_score(eval_labels_ss, clf.predict(eval_data_ss)))

    print(f"score: {accuracy_score(eval_labels_ss, clf.predict(eval_data_ss))}")

    x0 = [0]*patch_size*patch_size
    # iterations = [100, 200, 500, 1000]
    # epsilons = [0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 1.0]
    epsilons = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]


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
                    #print(f"current fitness: {curr_fitness}")
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

            #print(f"noise norm: {noise_norm}")
            correct_label.append(clf.predict(image.reshape(1,-1))[0])
            patch_image = add_patch_to_image(image, res['x'], patch_size, row_opt, col_opt)
            noise_label.append(clf.predict((patch_image).reshape(1,-1))[0])


            if correct_label[-1] != noise_label[-1]:
                fig = plt.figure()
                rows = 1
                columns = 3
                fig.text(0.15, 0.05, f"original label: {correct_label[-1]}")
                fig.text(0.4, 0.05, f"adversarial label: {noise_label[-1]}")
                fig.text(0.275, 0.15, f"real class: {eval_labels_ss[1:100][image_no]}")
                fig.add_subplot(1, 3, 1)
                plt.imshow(image.reshape([28,28]), cmap=plt.get_cmap('gray_r'))
                fig.add_subplot(1, 3, 2)
                plt.imshow(patch_image.reshape([28, 28]), cmap=plt.get_cmap('gray_r'))
                fig.add_subplot(1, 3, 3)
                plt.imshow(np.array(res['x']).reshape([3, 3]), cmap=plt.get_cmap('gray_r'))
                plt_name = f'one_iter_faster/est{est}_epsilon{epsilon}_image{image_no}.png'
                plt.savefig(plt_name)


        mean_fitness.append(np.mean(optimal_fitness)*(-1))
        mean_noise.append(np.mean(optimal_noise))
        min_noise.append(np.min(optimal_noise))
        max_noise.append(np.max(optimal_noise))
        var_noise.append(np.std(optimal_noise))

        mean_l0.append(np.mean(optimal_l0))
        min_l0.append(np.min(optimal_l0))
        max_l0.append(np.max(optimal_l0))
        var_l0.append(np.std(optimal_l0))


        mean_l1.append(np.mean(optimal_l1))
        min_l1.append(np.min(optimal_l1))
        max_l1.append(np.max(optimal_l1))
        var_l1.append(np.std(optimal_l1))

        correct_label = np.array(correct_label)
        noise_label = np.array(noise_label)

        wrong_output.append((len(correct_label[correct_label!=noise_label])/len(correct_label))*100)

    mean_fitness = np.array(mean_fitness).reshape(-1,1)
    mean_noise = np.array(mean_noise).reshape(-1,1)
    var_noise = np.array(var_noise).reshape(-1,1)
    min_noise = np.array(min_noise).reshape(-1,1)
    max_noise = np.array(max_noise).reshape(-1,1)
    mean_l0 = np.array(mean_l0).reshape(-1, 1)
    min_l0 = np.array(min_l0).reshape(-1, 1)
    max_l0 = np.array(max_l0).reshape(-1, 1)
    var_l0 = np.array(var_l0).reshape(-1, 1)
    mean_l1 = np.array(mean_l1).reshape(-1, 1)
    min_l1 = np.array(min_l1).reshape(-1, 1)
    max_l1 = np.array(max_l1).reshape(-1, 1)
    var_l1 = np.array(var_l1).reshape(-1, 1)
    wrong_output = np.array(wrong_output).reshape(-1,1)

    data = np.concatenate((mean_fitness, mean_noise, var_noise, min_noise, max_noise, wrong_output, mean_l0, min_l0, max_l0, var_l0, mean_l1, min_l1, max_l1, var_l1), axis = 1)
	


    #Writing data to file
    file = 'one_iter_faster/data' + str(est) + '.csv'
    with open(file, 'w', newline = '') as output:
        writer = csv.writer(output, delimiter=',')
        writer.writerows(data)

    #Results
    f1, ax1 = plt.subplots()
    ax1.plot(epsilons, mean_fitness)
    plt.title('Mean Optimal fitness vs epsilon')
    plt.xlabel('Epsilon')
    plt.ylabel('Mean Optimal fitness values')
    plt_name=  'one_iter_faster/plot_fitness'+str(est) +'.png'
    plt.savefig(plt_name)

    f2, ax2 = plt.subplots()
    ax2.plot(epsilons, mean_noise)
    plt.title('Mean Optimal noise vs epsilon')
    plt.xlabel('Epsilon')
    plt.ylabel('Mean Optimal noise')
    plt_name = 'one_iter_faster/plot_noise' + str(est) + '.png'
    plt.savefig(plt_name)

    f3, ax3 = plt.subplots()
    ax3.plot(epsilons, mean_noise, 'k-')
    ax3.fill_between(epsilons, list(np.ndarray.flatten(np.array(mean_noise) + np.array(var_noise))), list(np.ndarray.flatten(np.array(mean_noise) - np.array(var_noise))))
    plt.title('Optimal noise distribution vs epsilon')
    plt.xlabel('Epsilon')
    plt.ylabel('Optimal Noise Distribution')
    plt_name = 'one_iter_faster/plot_distribution' + str(est) + '.png'
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
    plt_name = 'one_iter_faster/plot_min' + str(est) + '.png'
    plt.savefig(plt_name)

    f5, ax5 = plt.subplots()
    ax5.plot(epsilons, max_noise)
    plt.title('Max noise vs epsilon')
    plt.xlabel('Epsilon')
    plt.ylabel('Max Noise')
    plt_name = 'one_iter_faster/plot_max' + str(est) + '.png'
    plt.savefig(plt_name)

    f6, ax6 = plt.subplots()
    ax6.plot(epsilons, wrong_output)
    plt.title('Percentage misclassified vs epsilon')
    plt.xlabel('Epsilon')
    plt.ylabel('Percentage misclassified')
    plt_name = 'one_iter_faster/plot_misclassified' + str(est) + '.png'
    plt.savefig(plt_name)

    #L_0 norm
    f7, ax7 = plt.subplots()
    ax7.plot(epsilons, mean_l0)
    plt.title('Mean Optimal L0 norm vs epsilon')
    plt.xlabel('Epsilon')
    plt.ylabel('Mean L0 norm')
    plt_name = 'one_iter_faster/l0_' + str(est) + '.png'
    plt.savefig(plt_name)

    f8, ax8 = plt.subplots()
    ax8.plot(epsilons, min_l0)
    plt.title('Min L0 norm vs epsilon')
    plt.xlabel('Epsilon')
    plt.ylabel('Min L0 norm')
    plt_name = 'one_iter_faster/plot_min_l0_' + str(est) + '.png'
    plt.savefig(plt_name)

    f9, ax9 = plt.subplots()
    ax9.plot(epsilons, max_l0)
    plt.title('Max L0 norm vs epsilon')
    plt.xlabel('Epsilon')
    plt.ylabel('Max L0 norm')
    plt_name = 'one_iter_faster/plot_max_l0_' + str(est) + '.png'
    plt.savefig(plt_name)

    f10, ax10 = plt.subplots()
    ax10.plot(epsilons, mean_l0, 'k-')
    ax10.fill_between(epsilons, list(np.ndarray.flatten(np.array(mean_l0) + np.array(var_l0))), list(np.ndarray.flatten(np.array(mean_l0) - np.array(var_l0))))
    plt.title('L0 norm distribution vs epsilon')
    plt.xlabel('Epsilon')
    plt.ylabel('L0 norm')
    plt_name = 'one_iter_faster/plot_distribution_l0_' + str(est) + '.png'
    plt.savefig(plt_name)

    #L_1 norm
    f11, ax11 = plt.subplots()
    ax11.plot(epsilons, mean_l1)
    plt.title('Mean Optimal L1 norm vs epsilon')
    plt.xlabel('Epsilon')
    plt.ylabel('Mean L1 norm')
    plt_name = 'one_iter_faster/l1_' + str(est) + '.png'
    plt.savefig(plt_name)

    f12, ax12 = plt.subplots()
    ax12.plot(epsilons, min_l1)
    plt.title('Min L1 norm vs epsilon')
    plt.xlabel('Epsilon')
    plt.ylabel('Min L1 norm')
    plt_name = 'one_iter_faster/plot_min_l1_' + str(est) + '.png'
    plt.savefig(plt_name)

    f13, ax13 = plt.subplots()
    ax13.plot(epsilons, max_l1)
    plt.title('Max L1 norm vs epsilon')
    plt.xlabel('Epsilon')
    plt.ylabel('Max L1 norm')
    plt_name = 'one_iter_faster/plot_max_l1_' + str(est) + '.png'
    plt.savefig(plt_name)

    f14, ax14 = plt.subplots()
    ax14.plot(epsilons, mean_l1, 'k-')
    ax14.fill_between(epsilons, list(np.ndarray.flatten(np.array(mean_l1) + np.array(var_l1))), list(np.ndarray.flatten(np.array(mean_l1) - np.array(var_l1))))
    plt.title('L1 norm distribution vs epsilon')
    plt.xlabel('Epsilon')
    plt.ylabel('L1 norm')
    plt_name = 'one_iter_faster/plot_distribution_l1_' + str(est) + '.png'
    plt.savefig(plt_name)



accuracy = np.array(accuracy).reshape(-1,1)
file = 'one_iter_faster/accuracy.csv'
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

	



