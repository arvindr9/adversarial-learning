from scripts.adv_gen.adv_gen import AdversarialGeneration
from scripts.adv_gen.process import ProcessData
from scripts.adv_gen.adv_test import AdversarialTest
from scripts.adv_gen.adv_transfer import AdversarialTransfer
from scripts.adv_gen.transfer_plot import TransferPlot
from scripts.adv_train.adv_train import AdversarialBoost
from scripts.adv_train.adv_train_process import TrainProcess
from scripts.adv_train.adv_train_plot import TrainPlot
import os
import argparse
import time
# import adv_test
# import adv_transfer
# import adv_train


#time

#Run script

#data paths
base_classifier = 'random_forest'
data_path = raw_data_path = os.getcwd() + '/data/adv_gen/rf/raw_data'
processed_data_path = os.getcwd() + '/data/adv_gen/rf/processed_data'
plot_path = os.getcwd() + '/data/adv_gen/rf/plots'
transfer_data_path = os.getcwd() + '/data/adv_gen/rf/transfer_data'
transfer_plot_path = os.getcwd() + '/data/adv_gen/rf/transfer_plots'
train_data_path = os.getcwd() + '/data/adv_train/rf'
train_clfs_path = os.getcwd() + '/data/adv_train/rf/clfs'


#generation params
max_depth = 10
criterion = 'entropy'
no_adv_images = 100
n_estimators = [1, 2, 5, 10, 20, 50, 100]
epsilons = [0.0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
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

#generation arguments
parser.add_argument("--gen", default = 0, help="whether to run the generation script", type = int)
parser.add_argument("--process", default = 0, help="whether to process data", type = int)
parser.add_argument("--plot", default = 0, help ="whether to plot the data", type =int)
parser.add_argument("--train_size", default = train_size, help="size of the subset of the training set", type = int)
parser.add_argument("--gen_adv_images", default = no_adv_images, help="number of adversarial images to create in the adversarial generation", type = int)

#transfer arguments
parser.add_argument("--transfer", default = 0, help="whether to run the script to process transferability data", type = int)
parser.add_argument("--transfer_plot", default = 0, help="whether to run the script to plot transferability data", type = int)

#train arguments
parser.add_argument("--train_threads", default = train_no_threads, help = "number of threads for adversarial training", type = int)
parser.add_argument("--train_estimators", default = train_estimators, help="number of estimators of the random forest to be trained", type = int)
parser.add_argument("--train", default = 0, help="whether or not to run the train script", type = int)
parser.add_argument("--adv_size", default = n_boost_random_train_samples, help = "number of adversarial images that are generated in each training iteration", type = int)
parser.add_argument("--boosting_iter", default = boosting_iter, help = "number of iterations of the boosting algorithm", type = int)
parser.add_argument("--train_process", default = 0, help = "whether to process the adversarially trained clf", type = int)
parser.add_argument("--train_plot", default = 0, help = "whether to plot the data from the training phase", type = int)

parser.add_argument("--file_type", default = file_type, help = "format to save images", type = str)

args = parser.parse_args()
arguments = args.__dict__

#Make a dictionary of all arguments
args_dict = {k: v for k,v in arguments.items()}


train_size = args_dict['train_size']

n_boost_random_train_samples = args_dict['adv_size']
train_estimators = args_dict['train_estimators']
train_no_threads = args_dict['train_threads']
boosting_iter = args_dict['boosting_iter']

file_type = args_dict['file_type']
no_adv_images = args_dict['gen_adv_images']


rf_params = {
    'max_depth': max_depth,
    'criterion': criterion,
    'n_estimators': n_estimators
}





gen_config = {
    'base_classifier': base_classifier,
    'data_path': data_path,
    'max_depth': max_depth,
    'criterion': criterion,
    'no_adv_images': no_adv_images,
    'n_estimators': n_estimators,
    'epsilons': epsilons,
    'inner_iter': inner_iter,
    'outer_iter': outer_iter,
    'no_threads': no_threads
}

#Certain global variables should not be shared


if args_dict['gen']:
    t_start = time.time()
    AdversarialGeneration(config = gen_config)
    t_end = time.time()
    with open(raw_data_path + "/time", "w") as f:
        f.write("Classifier: {}\n".format(gen_config['base_classifier']))
        f.write("Number of examples: {}\n".format(gen_config['no_adv_images']))
        f.write("Epsilon values: {}\n".format(gen_config['epsilons']))
        f.write("Param values: {}\n".format(gen_config['n_estimators']))
        f.write("Inner iterations: {}\n".format(gen_config['inner_iter']))
        f.write("Outer iterations: {}\n".format(gen_config['outer_iter']))
        f.write("Training set size: {}\n".format(gen_config['train_size']))
        f.write("Adversarial genaration time: {} seconds".format(t_end - t_start))

# Process script

process_config = {
    'base_classifier': base_classifier,
    'raw_data_path': raw_data_path,
    'processed_data_path': processed_data_path,
    'no_adv_images': no_adv_images,
    'n_estimators': n_estimators,
    'epsilons': epsilons
}

if args_dict['process']: ProcessData(config = process_config)

test_config = {
    'base_classifier': base_classifier,
    'processed_data_path': processed_data_path,
    'plot_path': plot_path,
    'epsilons': epsilons,
    'n_estimators': n_estimators,
    'file_type': file_type
}

if args_dict['plot']: AdversarialTest(config = test_config)

transfer_config = {
    'base_classifier': base_classifier,
    'raw_data_path': raw_data_path,
    'transfer_data_path': transfer_data_path,
    'no_adv_images': no_adv_images,
    'n_estimators': n_estimators,
    'epsilons': epsilons
}

if args_dict['transfer']: AdversarialTransfer(config = transfer_config)

transfer_plot_config = {
    'base_classifier': base_classifier,
    'transfer_data_path': transfer_data_path,
    'transfer_plot_path': transfer_plot_path,
    'n_estimators': n_estimators,
    'no_adv_images': no_adv_images,
    'epsilons': epsilons,
    'file_type': file_type
}

if args_dict['transfer_plot']: TransferPlot(config = transfer_plot_config)

train_config = {
    'base_classifier': base_classifier,
    'train_size': boosting_train_size,
    'boosting_iter': boosting_iter,
    'n_boost_random_train_samples': n_boost_random_train_samples,
    'estimator_params': {'n_estimators': train_estimators, 'criterion': 'entropy', 'max_depth': train_max_depth, 'min_samples_split': min_samples_split},
    'train_data_path': train_data_path,
    'epsilon': train_epsilon,
    'inner_iter': train_inner_iter,
    'outer_iter': train_outer_iter,
    'no_threads': train_no_threads 
}


if args_dict['train']:
    t_start = time.time()
    ab = AdversarialBoost(config = train_config)
    ab.main()
    t_end = time.time()
    t_diff = t_end - t_start
    with open(train_data_path + '/time', 'w') as f:
        print("Adversarial train time:", t_diff)
        print("Parameters:")
        print(train_config)


train_process_config = {
    'base_classifier': base_classifier,
    'boosting_iter': boosting_iter,
    'no_threads': train_no_threads,
    'train_data_path': train_data_path,
    'n_test_adv_images': n_test_adv_images,
    'train_estimators': train_estimators,
    'epsilons': epsilons
}

if args_dict['train_process']:
    TrainProcess(config = train_process_config)

train_plot_config = {
    'base_classifier': base_classifier,
    'boosting_iter': boosting_iter,
    'no_threads': train_no_threads,
    'train_data_path': train_data_path,
    'n_test_adv_images': n_test_adv_images,
    'train_estimators': train_estimators,
    'epsilons': epsilons
}

if args_dict['train_plot']:
    TrainPlot(config = train_plot_config)

