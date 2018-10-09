##Parameters
Classifier: Random Forest Classifier
Size of training set: 50000
Size of test set for adversarial image generation: 500

#Estimator parameters:
Spiltting criterion: ‘entropy’, Max depth: 10
Number of base estimators: [1, 2, 5, 10, 20, 50, 100]

#Optimization parameters
Epsilon values: [0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
Optimization method: Basinhopping (10 iterations) with an inner loop of SLSQP (100 SLSQP iterations per basinhopping iteration)

