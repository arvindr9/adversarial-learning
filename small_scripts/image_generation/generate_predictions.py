import csv
import cPickle
from numpy import genfromtxt

gen_rf_path = "../../data/adv_gen/rf"
raw_data_path = gen_rf_path + "/raw_data"

n_images = 500
epsilons = [0.001, 0.01, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
n_estimators = 50

clean_path = raw_data_path + '/images'
noise_path = raw_data_path + '/noise'
correct_labels_path = raw_data_path + '/correct_labels'

dest_path = 'image_data'
images_path = dest_path + '/images'
predictions_path = dest_path + '/predictions'

clfs = None

with open(raw_data_path + '/clfs.pkl') as f:
    clfs = cPickle.load(f)

clf = clfs[n_estimators]

for epsilon in epsilons:

    image_file = '/n_estimators_' + str(n_estimators) + '_eps_' + str(epsilon) + '.csv'
    images = genfromtxt(clean_path + image_file, delimiter= ',')
    noise = genfromtxt(noise_path + image_file, delimiter= ',')
    adv_images = images + noise

    # print(images[0])

    with open(images_path + '/clean' + image_file, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(images)
    with open(images_path + '/adv' + image_file, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(adv_images)

    clean_predictions = clf.predict(images)
    adv_predictions = clf.predict(adv_images)
    correct_labels = genfromtxt(correct_labels_path + image_file, delimiter = ',')

    with open(predictions_path + '/clean' + image_file, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(clean_predictions)
    with open(predictions_path + '/adv' + image_file, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(adv_predictions)
    with open(predictions_path + '/correct_labels' + image_file, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(correct_labels)
