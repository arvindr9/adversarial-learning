from scripts.adv_train_random.adv_train_random import AdvTrainRandom
from scripts.adv_train_random.random_process import RandomProcess
from scripts.adv_train_random.random_plot import RandomPlot

import os
import argparse
import time


base_classifier = 'random_forest'
random_data_path = os.getcwd() + '/data/random_noise/rf'


#data paths
base_classifier = 'random_forest'
data_path = raw_data_path = os.getcwd() + '/data/adv_gen/rf/raw_data'
processed_data_path = os.getcwd() + '/data/adv_gen/rf/processed_data'
plot_path = os.getcwd() + '/data/adv_gen/rf/plots'
transfer_data_path = os.getcwd() + '/data/adv_gen/rf/transfer_data'
transfer_plot_path = os.getcwd() + '/data/adv_gen/rf/transfer_plots'
train_data_path = os.getcwd() + '/data/adv_train/rf'
train_clfs_path = os.getcwd() + '/data/adv_train/rf/clfs'
random_data_path = os.getcwd() + '/data/random_noise/rf'
random_clfs_path = os.getcwd() + '/data/random_noise/rf/clfs'



#generation params
max_depth = 10
criterion = 'entropy'
no_adv_images = 500
n_estimators = [1, 2, 5, 10, 20, 50, 100]
epsilons = [0.0, 0.001, 0.01, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
inner_iter = 20
outer_iter = 5
no_threads = 20
train_size = 50000

#training params
train_no_threads = no_threads
train_inner_iter = inner_iter
train_outer_iter = outer_iter
train_epsilon = 0.3
train_max_depth = 5
boosting_train_size = 5000
boosting_iter = 100
train_estimators = 5
n_boost_random_train_samples = 500
min_samples_split = 5
train_data_path = os.getcwd() + '/data/adv_train/rf'

n_test_adv_images = 200

file_type = 'pdf'

parser = argparse.ArgumentParser()


#random arguments
parser.add_argument("--random_noise", default = 0, help = "whether to run the experiment that adds random noise to the images")
parser.add_argument("--random_process", default = 0, help = "whether to process the random data")
parser.add_argument("--random_plot", default = 0, help = "whether to plot the random data")

parser.add_argument("--random_estimators", default = 5, help = "number of estimators for the random script", type = int)

parser.add_argument("--file_type", default = file_type, help = "format to save images", type = str)

args = parser.parse_args()
arguments = args.__dict__

#Make a dictionary of all arguments
args_dict = {k: v for k,v in arguments.items()}

file_type = args_dict['file_type']
# no_adv_images = args_dict['gen_adv_images']
random_estimators = args_dict['random_estimators']


random_config = {
    'base_classifier': base_classifier,
    'train_size': boosting_train_size,
    'boosting_iter': boosting_iter,
    'n_boost_random_train_samples': n_boost_random_train_samples,
    'estimator_params': {'n_estimators': random_estimators, 'criterion': 'entropy', 'max_depth': train_max_depth, 'min_samples_split': min_samples_split},
    'random_data_path': random_data_path,
    'epsilon': train_epsilon,
    'inner_iter': train_inner_iter,
    'outer_iter': train_outer_iter,
}

if args_dict['random_noise']:
    print("About to run")
    ab = AdvTrainRandom(config = random_config)
    ab.main()
    print("Finished running")

process_config = {
    'random_data_path': random_data_path,
    'n_test_adv_images': n_test_adv_images,
    'boosting_iter': boosting_iter,
    'random_estimators': random_estimators,
    'epsilons': epsilons,
    'no_threads': no_threads
}

if args_dict['random_process']:
    print("About to process")
    RandomProcess(config = process_config)
    print("Finished processing")

plot_config = {
    'epsilons': epsilons,
    'gen_data_path': processed_data_path,
    'random_data_path': random_data_path,
    'random_estimators': random_estimators,
    'gen_estimators': n_estimators,
    'boosting_iter': boosting_iter

}

if args_dict['random_plot']:
    print("About to plot")
    RandomPlot(config = plot_config)
    print("Done plotting")