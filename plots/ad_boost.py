import numpy as np
import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from scipy.optimize import basinhopping
from scipy.linalg import norm
import matplotlib.pyplot as plt



class AdversarialBoost:

    def __init__(self, n_estimators, learning_rate, to_add_adv, epsilon, n_adv, n_boost_random_train_samples):

        self.n_estimators = n_estimators
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(n_estimators, dtype= np.float64)
        self.learning_rate = learning_rate
        self.to_add_adv = to_add_adv
        self.epsilon = epsilon
        self.n_adv = n_adv
        self.n_boost_random_train_samples = n_boost_random_train_samples


    def fit(self, X, y):

        sample_weight = np.empty(X.shape[0], dtype=np.float64)
        sample_weight[:] = 1. / X.shape[0]
        


        for iboost in range(self.n_estimators):

            print("Boosting step : {}".format(iboost))


            # Boosting step
            cur_estimator, sample_weight, estimator_weight, estimator_error = self._boost(iboost, X, y, sample_weight)

            if(self.to_add_adv == 1):

                #Generating adversarial examples

                print("Generating adversarial examples.....")


                adv_examples, _ = self._advGen(cur_estimator, self.n_adv, self.n_boost_random_train_samples, self.epsilon, X, y)


                print("Done with Generating adversarial examples.....")
                #Replacing the best performing normal samples with best performing adversarial examples
                if iboost == 0:

                    #If sample weight is uniform then just replace any normal samples with adversarial examples
                    X[:self.n_adv,:] = adv_examples 

                else:
                    
                    #If sample weight is not uniform then replace the n_adv normal examples with least sample weight
                    index_sorted_samples = sorted(range(len(sample_weight)), key = lambda k: sample_weight[k])
                    X[index_sorted_samples[:self.n_adv], :] = adv_examples

            # Early termination
            if sample_weight is None:
                break

            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            # Stop if error is zero
            if estimator_error == 0:
                break

            sample_weight_sum = np.sum(sample_weight)

            # Stop if the sum of sample weights has become non-positive
            if sample_weight_sum <= 0:
                break

            if iboost < self.n_estimators - 1:
                # Normalize
                sample_weight /= sample_weight_sum

        return self



    def _boost(self, iboost, X, y, sample_weight):


            estimator = DecisionTreeClassifier( criterion ='gini', max_depth = 10)

            estimator.fit(X, y, sample_weight=sample_weight)

            self.estimators_.append(estimator)

            y_predict = estimator.predict(X)

            if iboost == 0:
                self.classes_ = getattr(estimator, 'classes_', None)
                self.n_classes_ = len(self.classes_)

            # Instances incorrectly classified
            incorrect = y_predict != y

            # Error fraction
            estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))

            # Stop if classification is perfect
            if estimator_error <= 0:
                return sample_weight, 1., 0.

            n_classes = self.n_classes_
            # Stop if the error is at least as bad as random guessing
            if estimator_error >= 1. - (1. / n_classes):
                self.estimators_.pop(-1)
                if len(self.estimators_) == 0:
                    raise ValueError('BaseClassifier in Adversarial Boost classifier '
                                     'ensemble is worse than random, ensemble '
                                     'can not be fit.')
                return None, None, None

            estimator_weight = self.learning_rate * (
                np.log((1. - estimator_error) / estimator_error) +
                np.log(n_classes - 1.))

            # Only boost the weights if I will fit again
            if not iboost == self.n_estimators - 1:
                # Only boost positive weights
                sample_weight *= np.exp(estimator_weight * incorrect *
                                        ((sample_weight > 0) |
                                         (estimator_weight < 0)))

            return estimator, sample_weight, estimator_weight, estimator_error

    def predict(self, X):
        """Predict classes for X.
        The predicted class of an input sample is computed as the weighted mean
        prediction of the classifiers in the ensemble.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.
        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes.
        """
        pred = self.decision_function(X)

        if self.n_classes_ == 2:
            return self.classes_.take(pred > 0, axis=0)

        return self.classes_.take(np.argmax(pred, axis=1), axis=0)


    def decision_function(self, X):
        """Compute the decision function of ``X``.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.
        Returns
        -------
        score : array, shape = [n_samples, k]
            The decision function of the input samples. The order of
            outputs is the same of that of the `classes_` attribute.
            Binary classification is a special cases with ``k == 1``,
            otherwise ``k==n_classes``. For binary classification,
            values closer to -1 or 1 mean more like the first or second
            class in ``classes_``, respectively.
        """
        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis] #Column vector
        pred = None

        pred = sum((estimator.predict(X) == classes).T * w for estimator, w in zip(self.estimators_, self.estimator_weights_))

        pred /= self.estimator_weights_.sum()
        if n_classes == 2:
            pred[:, 0] *= -1
            return pred.sum(axis=1)
        return pred

    def fitness_func(self, noise, x, clf):

        x = np.array(x).reshape(1,-1)
        return -1*norm(clf.predict_proba(x) - clf.predict_proba(x+noise))


    def _advGen(self, estimator, n_adv, n_boost_random_train_samples, epsilon, X, y):


        x0 = [0] * np.shape(X)[1]

        cons = ({'type': 'ineq', 'fun': lambda noise: epsilon**2 - norm(np.array(noise))})

        optimal_noise = []
        optimal_fitness = []

        rand_idx = np.random.randint(0,len(X), (n_boost_random_train_samples,))
        rand_X = X[rand_idx]
        rand_y = y[rand_idx]

        for image_no, image in enumerate(rand_X):
            
            print("Generating adversarial examples for image number : {}".format(image_no), end = "\r")

            minimizer_kwargs = dict(method = "slsqp", args = (image, estimator), constraints = cons, options = {'maxiter': 100})
            res = basinhopping(self.fitness_func, niter = 1, x0 = x0, minimizer_kwargs = minimizer_kwargs)
            optimal_fitness.append(res['fun'])
            cur_noise =res['x']
            
            optimal_noise.append(cur_noise)

        index_noise = sorted(range(len(optimal_fitness)), key=lambda k: optimal_fitness[k])

        adv_examples = []
        adv_y = []
        for index in index_noise[:n_adv]:
            adv_examples.append(rand_X[index, :] + optimal_noise[index])
            adv_y.append(rand_y[index])

        return np.array(adv_examples), adv_y

        

if(__name__ == '__main__'):


    #Loading Mnist data
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    #Subsampling the training dataset
    rand_idx = np.random.randint(0,len(train_data), (500,))
    train_data_ss = train_data[rand_idx]
    train_labels_ss = train_labels[rand_idx]

    #Subsampling the testing dataset
    rand_idx = np.random.randint(0,len(eval_data), (500,))
    eval_data_ss = eval_data[rand_idx]
    eval_labels_ss = eval_labels[rand_idx]

    no_boosting_clf = 10
    learning_rate = 1
    epsilon = 0.5
    n_advimages = 30
    n_boost_random_train_samples =  100


    #Adversarial Training

    ab = AdversarialBoost(no_boosting_clf, learning_rate, 1, epsilon, n_advimages, n_boost_random_train_samples)
    ab.fit(train_data_ss, train_labels_ss)

    print("Overall testing accuracy on normal images with adversarial training: {}".format(accuracy_score(eval_labels_ss, ab.predict(eval_data_ss))))

    
    #Normal boosting without adversarial examples
    ab_n = AdversarialBoost(no_boosting_clf, learning_rate, 0, epsilon, n_advimages, n_boost_random_train_samples)
    ab_n.fit(train_data_ss, train_labels_ss)

    print("Overall testing accuracy on normal images without adversarial training: {}".format(accuracy_score(eval_labels_ss, ab_n.predict(eval_data_ss))))
    
    #Generating some adversarial examples
    #Plotting accuracy of each weak classifier on both normal test images and adversarial images

    noise_accuracy_adv = []
    normal_accuracy_adv = []

    noise_accuracy_norm = []
    normal_accuracy_norm = []

    estimator_no = 1

    for est_adv, est_norm in zip(ab.estimators_, ab_n.estimators_):

        print('Estimator number: {}'.format(estimator_no))

        adv_examples_test_adv, adv_true_adv = ab._advGen(est_adv, 30, 100, epsilon, eval_data_ss, eval_labels_ss)
        adv_examples_test_norm, adv_true_norm = ab_n._advGen(est_norm, 30, 100, epsilon, eval_data_ss, eval_labels_ss)

        noise_accuracy_adv.append(accuracy_score(adv_true_adv, est_adv.predict(adv_examples_test_adv)))
        normal_accuracy_adv.append(accuracy_score(eval_labels_ss, est_adv.predict(eval_data_ss)))

        noise_accuracy_norm.append(accuracy_score(adv_true_norm, est_norm.predict(adv_examples_test_norm)))
        normal_accuracy_norm.append(accuracy_score(eval_labels_ss, est_norm.predict(eval_data_ss)))

        estimator_no +=1



    fig1, ax1 = plt.subplots()
    plt.plot(list(range(len(ab.estimator_errors_))),ab.estimator_errors_, linewidth = 2, color = 'red')
    plt.plot(list(range(len(ab_n.estimator_errors_))), ab_n.estimator_errors_, linewidth =2, color = 'blue')
    plt.legend(['With adversarial Training', 'Without adversarial Training'])
    plt.ylabel('Error')
    plt.xlabel('Boosting Step')
    plt.title('Error for each of the weak classifier with/without adversarial training')
    plt.savefig('Error.png')

    fig2, ax2 = plt.subplots()
    plt.plot(list(range(len(ab.estimator_weights_))),ab.estimator_weights_, linewidth = 2, color = 'red')
    plt.plot(list(range(len(ab_n.estimator_weights_))), ab_n.estimator_weights_, linewidth =2, color = 'blue')
    plt.legend(['With adversarial Training', 'Without adversarial Training'])
    plt.ylabel('Weights')
    plt.xlabel('Boosting Step')
    plt.title('Weights for each of the weak classifier with/without adversarial training')
    plt.savefig('weights.png')


    fig3, ax3 = plt.subplots()
    plt.plot(list(range(len(normal_accuracy_adv))), normal_accuracy_adv, linewidth =2, color = 'red')
    plt.plot(list(range(len(normal_accuracy_norm))), normal_accuracy_norm, linewidth =2, color = 'blue')
    plt.legend(['With adversarial Training', 'Without adversarial Training'])
    plt.ylabel('Accuracy')
    plt.xlabel('Boosting Estimators')
    plt.title('Accuracy for normal test images')
    plt.savefig('normal_accuracy.png')

    fig4, ax4 = plt.subplots()
    plt.plot(list(range(len(noise_accuracy_adv))), noise_accuracy_adv, linewidth =2, color = 'red')
    plt.plot(list(range(len(noise_accuracy_norm))), noise_accuracy_norm, linewidth =2, color = 'blue')
    plt.legend(['With adversarial Training', 'Without adversarial Training'])
    plt.ylabel('Accuracy')
    plt.xlabel('Boosting Estimators')
    plt.title('Accuracy for adversarial test images (generated by corresponding estimator)')
    plt.savefig('adv_accuracy.png')

