import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt

images_path = 'image_data/images'
predictions_path = 'image_data/predictions'
dest_path = 'plots'

epsilons = [0.001, 0.01, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
n_estimators = 50
n_images = 50

for epsilon in epsilons:
    image_file = '/n_estimators_' + str(n_estimators) + '_eps_' + str(epsilon) + '.csv'
    clean_images = genfromtxt(images_path + '/clean' + image_file, delimiter = ',')
    adv_images = genfromtxt(images_path + '/adv' + image_file, delimiter = ',')
    clean_predictions = genfromtxt(predictions_path + '/clean' + image_file, delimiter= ',')
    adv_predictions = genfromtxt(predictions_path + '/adv' + image_file, delimiter= ',')
    correct_labels = genfromtxt(predictions_path + '/correct_labels' + image_file, delimiter= ',')
    print(clean_images.shape)
    for i in range(n_images):
        clean_image = clean_images[i]
        adv_image = adv_images[i]
        clean_prediction = int(clean_predictions[i])
        adv_prediction = int(adv_predictions[i])
        correct_label = correct_labels[i]
        if clean_prediction == correct_label and clean_prediction != adv_prediction:
            fig = plt.figure()
            fig.text(0.43, 0.9, "Epsilon: " + str(epsilon))
            fig.text(0.2, 0.8, "Clean image")
            fig.text(0.65, 0.8, "Adversarial image")
            fig.text(0.6, 0.15, "Adversarial prediction: " +  str(int(adv_prediction)))
            fig.text(0.45, 0.08,"Correct label: " + str(int(correct_label)))
            fig.text(0.15, 0.15, "Clean prediction: " +  str(int(clean_prediction)))
            fig.add_subplot(1, 2, 1)
            plt.imshow(clean_image.reshape(28, 28), cmap=plt.get_cmap('gray_r'))
            fig.add_subplot(1, 2, 2)
            plt.imshow(adv_image.reshape(28, 28), cmap=plt.get_cmap('gray_r'))
            plt.savefig("plots/n_estimators_" + str(n_estimators) + '_eps_' + str(epsilon) + "_image_" + str(i) + ".pdf")
























