from scripts.adv_gen_cifar.adv_gen_cifar import AdversarialGeneration
import os
import argparse
import time




base_classifier = 'random_forest'
home_path = os.getcwd()
data_path = raw_data_path = os.getcwd() + '/data/adv_gen_cifar/rf/raw_data'
processed_data_path = os.getcwd() + '/data/adv_gen_cifar/rf/processed_data'
plot_path = os.getcwd() + '/data/adv_gen_cifar/rf/plots'

#generation params
max_depth = 10
criterion = 'entropy'
no_adv_images = 100
n_estimators = [1, 2, 5]#, 10, 20, 50, 100] #CHANGE
epsilons = [0.0, 0.01, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0] #CHANGE #[0.0, 0.001, 0.01, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
inner_iter = 20
outer_iter = 5
no_threads = 5#20 #CHANGE
train_size = 50000


file_type = 'pdf'

parser = argparse.ArgumentParser()

#generation arguments
parser.add_argument("--gen", default = 0, help="whether to run the generation script", type = int)
parser.add_argument("--process", default = 0, help="whether to process data", type = int)
parser.add_argument("--plot", default = 0, help ="whether to plot the data", type =int)
parser.add_argument("--train_size", default = train_size, help="size of the subset of the training set", type = int)
parser.add_argument("--gen_adv_images", default = no_adv_images, help="number of adversarial images to create in the adversarial generation", type = int)
parser.add_argument("--gen_threads", default = no_threads, help = "number of threads for generation", type = int)


parser.add_argument("--file_type", default = file_type, help = "format to save images", type = str)

args = parser.parse_args()
arguments = args.__dict__

#Make a dictionary of all arguments
args_dict = {k: v for k,v in arguments.items()}


train_size = args_dict['train_size']
no_threads = args_dict['gen_threads']

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
    'no_threads': no_threads,
    'home_path': home_path
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
