# adversarial-learning

##Algorithm

* **Training size**: 50000 MNIST Images
* **Classifier**: Random Forest Classifier
* **Hyperparameters**:
  * Splitting criteria: entropy
  * Max depth: 10
  * Number of estimators: varying (1, 2, 3, 5, 10). Note: Larger numbers of estimators will eventually be tested.
* **Optimization algorithm**: Basinhopping (10 iterations)
* **Optimization parameters**:
  * Method: SQSLP (100 iterations by default)
  * Bounds: L2 norm < epsilon
  * Initial noise: [0] * 784
  * Iterations: 100
* **Number of images**: 100
