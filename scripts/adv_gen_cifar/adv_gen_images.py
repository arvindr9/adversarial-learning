import numpy as np
from numpy import genfromtxt
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import cPickle


class GenImages:

    def __init__(self, config):
        self.epsilons = config['epsilons']
        self.raw_data_path = config['raw_data_path']
        self.image_dest_path = config['image_dest_path']
        self.n_adv_images = config['n_adv_images']
        self.main()



    def generateImages(self, clf, epsilon):
        n_estimators = clf.n_estimators
        file_ending = 'n_estimators_' + str(n_estimators) + '_eps_' + str(epsilon) + '.csv'
        image_file = self.raw_data_path + '/images/' + file_ending
        noise_file = self.raw_data_path + '/noise/' + file_ending
        images = genfromtxt(image_file, delimiter = ',')
        noises = genfromtxt(noise_file, delimiter = ',')
        adv_images = images + noises
        images = images.reshape(self.n_adv_images, 32, 32, 3)
        adv_images = adv_images.reshape(self.n_adv_images, 32, 32, 3)
        for i in range(self.n_adv_images):
            image = images[i,:,:,:]
            adv_image = adv_images[i,:,:,:]
            fig = plt.figure(figsize=(100, 50))
            sub = fig.add_subplot(2, 1, 1) #rows, cols, current
            sub.imshow(image)
            sub = fig.add_subplot(2, 1, 2)
            sub.imshow(adv_image)
            print("About to save image")
            plt.savefig(self.image_dest_path + '/images_est_{}_eps_{}_image_{}.pdf'.format(n_estimators, epsilon, i))
            print("saved image")




    def main(self):

        #load clfs
        clf_file = open(self.raw_data_path + '/clfs.pkl')
        clfs = cPickle.load(clf_file)
        clf_file.close()

        for n_est in clfs:
            for epsilon in self.epsilons:
                self.generateImages(clfs[n_est], epsilon)