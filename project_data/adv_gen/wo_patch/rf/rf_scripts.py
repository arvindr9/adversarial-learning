import adv_gen
import adv_test
import adv_transfer
import adv_train


'''
import adv_test
import adv_train
time
'''

#Run script
base_classifier = 'random_forest'
data_path = 'raw_data'
max_depth = 10
criterion = 'entropy'
no_adv_images = 500
n_estimators = [1, 2, 5, 10, 20, 50, 100]
epsilons = [0.0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
inner_iter = 20
outer_iter = 5
no_threads = 2 #CHANGE





gen_config = {
    'base_classifier': base_classifier,
    'data_path': data_path
    'max_depth': max_depth
    'criterion': criterion,
    'no_adv_images': no_adv_images,
    'n_estimators': n_estimators,
    'epsilons': epsilons,
    'inner_iter': inner_iter,
    'outer_iter': outer_iter,
    'no_threads': no_threads
}

#Certain global variables should not be shared

generation = adv_gen(config = gen_config)



#Test script

test_config = {
    'base_classifier': base_classifier,
    'data_path': data_path
    'max_depth': max_depth
    'criterion': criterion,
    'no_adv_images': no_adv_images,
    'n_estimators': n_estimators,
    'epsilons': epsilons,
    'inner_iter': inner_iter,
    'outer_iter': outer_iter,
    'no_threads': no_threads
}
# 7 * (7 * 9) ~=~ 500 accuracy calculations
for est_i in n_estimators:
    for est_j in n_estimators:
        for eps in epsilons:
            adv_test(config = test_config, clf = est_i, data = est_j, epsilon = eps)


#fixed epsilon: run for each (est_i, est_j)


transfer_config = {
    'base_classifier': base_classifier,
    'data_path': data_path
    'max_depth': max_depth
    'criterion': criterion,
    'no_adv_images': no_adv_images,
    'n_estimators': n_estimators,
    'epsilons': epsilons,
    'inner_iter': inner_iter,
    'outer_iter': outer_iter,
    'no_threads': no_threads
}
# 7 * (7 * 9) ~=~ 500 accuracy calculations
for est_i in n_estimators:
    for est_j in n_estimators:
        for eps in epsilons:
            adv_transfer(config = transfer_config, clf = est_i, data = est_j, epsilon = eps)


#Training script

epsilon = 0.3

train_config = {
    'base_classifier': base_classifier,
    'data_path': data_path
    'max_depth': max_depth
    'criterion': criterion,
    'no_adv_images': no_adv_images,
    'n_estimators': n_estimators,
    'inner_iter': inner_iter,
    'outer_iter': outer_iter,
    'no_threads': no_threads
}

adv_train(config = train_config, epsilon = 0.3)


