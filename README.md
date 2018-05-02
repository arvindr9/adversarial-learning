# adversarial-learning

## About this repository

This is a project to generate adversarial examples to fool a Random Forest Classifier
and to propose a method to adversarially train Random Forest Classifiers.

A link to the results can be seen [here](https://docs.google.com/presentation/d/1kHk7nZ4z5ZOcIFnNr7NYfRCe-lZL5VdOIFdNz4p8TP4/edit?usp=sharing)

## Parameters for generation of adversarial examples

* **Training size**: 50000 MNIST Images
* **Classifier**: Random Forest Classifier
* **Hyperparameters**:
  * Splitting criteria: entropy
  * Max depth: 10
  * Number of estimators: varying (1, 2, 3, 5, 10). Note: Larger numbers of estimators will eventually be tested.
  * Epsilon: 0.1, 0.2, 0.3, 0.5, 0.8, 1.0
* **Optimization algorithm**: Basinhopping (10 iterations)
* **Optimization parameters**:
  * Method: SQSLP (100 iterations by default)
  * Bounds: L2 norm < epsilon
  * Initial noise: [0] * 784
  * Iterations: 100
* **Number of images**: 100

## Parameters for adversarial patching

* **Training size**: 50000 MNIST Images
* **Classifier**: Random Forest Classifier
* **Hyperparameters**:
  * Splitting criteria: entropy
  * Max depth: 10
  * Number of estimators: varying (1, 2, 3, 5, 10, 20, 50, 100)
  * Epsilon: 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.2

## Parameters f