from scripts.adv_gen.adv_gen import AdversarialGeneration
from scripts.adv_gen.process import ProcessData
import os
# import adv_test
# import adv_transfer
# import adv_train


#time

#Run script
base_classifier = 'random_forest'
data_path = raw_data_path = os.getcwd() + '/data/adv_gen/rf/raw_data'
processed_data_path = os.getcwd() + '/data/adv_gen/rf/processed_data'
max_depth = 10
criterion = 'entropy'
no_adv_images = 10 #CHANGE
n_estimators = [1, 2, 5, 10]#], 20, 50, 100] CHANGE
epsilons = [0.0, 0.001, 0.01]#], 0.1, 0.2, 0.3, 0.5, 0.7, 1.0] CHANGE
inner_iter = 20
outer_iter = 1 #CHANGE to 5
no_threads = 2

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

generation = AdversarialGeneration(config = gen_config)

# Process script

process_config = {
    'base_classifier': base_classifier,
    'raw_data_path': raw_data_path,
    'processed_data_path': processed_data_path,
    'no_adv_images': no_adv_images,
    'n_estimators': n_estimators,
    'epsilons': epsilons
}

process = ProcessData(config = process_config)

# #Test script

# test_config = {
#     'base_classifier': base_classifier,
#     'data_path': data_path
#     'max_depth': max_depth
#     'criterion': criterion,
#     'no_adv_images': no_adv_images,
#     'n_estimators': n_estimators,
#     'epsilons': epsilons,
#     'inner_iter': inner_iter,
#     'outer_iter': outer_iter,
#     'no_threads': no_threads
# }
# # 7 * (7 * 9) ~=~ 500 accuracy calculations
# for est_i in n_estimators:
#     for est_j in n_estimators:
#         for eps in epsilons:
#             adv_test(config = test_config, clf = est_i, data = est_j, epsilon = eps)


# #fixed epsilon: run for each (est_i, est_j)




# train_config = {
#     'base_classifier': base_classifier,
#     'data_path': data_path
#     'max_depth': max_depth
#     'criterion': criterion,
#     'no_adv_images': no_adv_images,
#     'n_estimators': n_estimators,
#     'inner_iter': inner_iter,
#     'outer_iter': outer_iter,
#     'no_threads': no_threads
# }

# adv_train(config = train_config, epsilon = 0.3)


