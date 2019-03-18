from scripts.adv_gen.adv_gen import AdversarialGeneration
from scripts.adv_gen.process import ProcessData
from scripts.adv_gen.adv_test import AdversarialTest
from scripts.adv_gen.adv_transfer import AdversarialTransfer
from scripts.adv_gen.transfer_plot import TransferPlot
from scripts.adv_train.adv_train import AdversarialBoost
import os
import argparse
import time
# import adv_test
# import adv_transfer
# import adv_train


#time

#Run script
base_classifier = 'svm'
data_path = raw_data_path = os.getcwd() + '/data/adv_gen/' + base_classifier + '/raw_data'
processed_data_path = os.getcwd() + '/data/adv_gen/' + base_classifier + '/processed_data'
plot_path = os.getcwd() + '/data/adv_gen/' + base_classifier + '/plots'
transfer_data_path = os.getcwd() + '/data/adv_gen/svm/transfer_data'
transfer_plot_path = os.getcwd() + '/data/adv_gen/svm/transfer_plots'
train_data_path = os.getcwd() + '/data/adv_train/svm'


no_adv_images = 100
C_vals = [.1, .2, 1, 2, 5, 10, 20, 50, 100] #add a gamma param?
epsilons = [0.0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
inner_iter = 20
outer_iter = 5
no_threads = 20
train_size = 5000

file_type = 'pdf'

parser = argparse.ArgumentParser()
#generation arguments
parser.add_argument("--gen", default = 0, help="whether to run the generation script", type = int)
parser.add_argument("--process", default = 0, help="whether to process data", type = int)
parser.add_argument("--plot", default = 0, help ="whether to plot the data", type =int)
parser.add_argument("--train_size", default = train_size, help="size of the subset of the training set", type = int)

#transfer arguments
parser.add_argument("--transfer", default = 0, help="whether to run the script to process transferability data", type = int)
parser.add_argument("--transfer_plot", default = 0, help="whether to run the script to plot transferability data", type = int)

parser.add_argument("--no_threads", default = no_threads, help = "number of threads to run the script with", type = int)

parser.add_argument("--file_type", default = file_type, help = "format to save images", type = str)

args = parser.parse_args()
arguments = args.__dict__

#Make a dictionary of all arguments
args_dict = {k: v for k,v in arguments.items()}

train_size = args_dict['train_size']
no_threads = args_dict['no_threads']

file_type = args_dict['file_type']


gen_config = {
    'base_classifier': base_classifier,
    'data_path': data_path,
    'no_adv_images': no_adv_images,
    'C_vals': C_vals,
    'epsilons': epsilons,
    'inner_iter': inner_iter,
    'outer_iter': outer_iter,
    'no_threads': no_threads,
    'train_size': train_size
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
        f.write("Param values: {}\n".format(gen_config['C_vals']))
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
    'C_vals': C_vals,
    'epsilons': epsilons
}

if args_dict['process']: ProcessData(config = process_config)

test_config = {
    'base_classifier': base_classifier,
    'processed_data_path': processed_data_path,
    'plot_path': plot_path,
    'epsilons': epsilons,
    'C_vals': C_vals,
    'file_type': file_type
}

if args_dict['plot']: AdversarialTest(config = test_config)






transfer_config = {
    'base_classifier': base_classifier,
    'raw_data_path': raw_data_path,
    'transfer_data_path': transfer_data_path,
    'no_adv_images': no_adv_images,
    'C_vals': C_vals,
    'epsilons': epsilons
}

if args_dict['transfer']: AdversarialTransfer(config = transfer_config)

transfer_plot_config = {
    'base_classifier': base_classifier,
    'transfer_data_path': transfer_data_path,
    'transfer_plot_path': transfer_plot_path,
    'C_vals': C_vals,
    'no_adv_images': no_adv_images,
    'epsilons': epsilons,
    'file_type': file_type
}

if args_dict['transfer_plot']: TransferPlot(config = transfer_plot_config)


