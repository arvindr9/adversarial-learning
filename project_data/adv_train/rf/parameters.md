##Parameters
classifier: Random Forest
training size: 10000
no_boosting_clf: 100
n_boost_random_train_samples: 1000

#Estimator parameters: 
n_estimators: 100, criterion: entropy, max_depth: 5, min_samples_split: 5

#Optimization parameters
Basinhopping: 10 iterations, inner optimization method: SLSQP (100 iterations)
SetSize: 20 #The number of adversarial images in each thread
epsilon_train: 0.3
