import pickle
import numpy as np

class AdversarialBoost:

	"""
	Class for Adversarial Boosting
	Parameters
	-----------
	base_estimator : scikit learn classifier
	n_estimators : number of estimators in boosting loop
	epsilon : (float) radius of noise for adversarial examples generation
	n_adv : (int) number of adversarial examples to train in each loop
	n_boost_random_train_samples : number of random examples to take from training data for generating adversarial examples
	estimator_params : estimator parameters
	"""

	def __init__(self, base_estimator, n_estimators, epsilon, n_adv, n_boost_random_train_samples, estimator_params = None):


		self.base_estimator = base_estimator
		self.n_estimators = n_estimators
		self.estimators_ = []
		self.estimator_errors_ = np.ones(n_estimators, dtype= np.float64)
		self.epsilon = epsilon
		self.n_adv = n_adv
		self.n_boost_random_train_samples = n_boost_random_train_samples
		self.estimator_params = estimator_params
		self.total_boosting_time = [] #Total time for each boosting step
		self.fitting_time = []	#Time for training one boosting classifier
		self.adv_generation_time = [] #Total time for generating all the adversarial images in each step
		self.concatenate_time = [] #Time for concatenating images
		self.per_image_advGen_time = [] #Time per image adversarial image generation


		if(isinstance(self.base_estimator, SVC)):
				
			self.base_estimator.set_params(**{'probability': True})

		if(isinstance(self.base_estimator, DecisionTreeClassifier)):

			self.base_estimator.set_params(**{'max_depth': 10})

		if(isinstance(self.base_estimator, RandomForestClassifier)):

			self.base_estimator.set_params(**{'max_depth': 10})

		if not hasattr(self.base_estimator, 'predict_proba'):
			raise TypeError("Base estimator has no attribute 'predict_proba'")


	def fit(self, X, y):

		for iboost in range(self.n_estimators):

			b_tot_start = time.time()


			b_fit_start = time.time()


			print "Boosting step : %d" %iboost


			# Training the weak classifier
			cur_estimator, estimator_error = self._boost(iboost, X, y)


			b_fit_end = time.time()

			#Generating adversarial examples

			b_adv_gen_start = time.time()

			print "Generating adversarial examples....." 


			adv_examples, adv_y = self._advGen(cur_estimator, self.n_adv, self.n_boost_random_train_samples, self.epsilon, X, y)


			b_adv_gen_end = time.time()


			print "\nDone with Generating adversarial examples....." 

			b_adv_conc_start = time.time()

			
			#Taking a union of adversarial examples and original training data
			X = np.concatenate((X, adv_examples), axis = 0)
			y = np.concatenate((y, adv_y), axis = 0)


			b_adv_conc_end = time.time()


			self.estimator_errors_[iboost] = estimator_error

			# Stop if error is zero
			# if estimator_error == 0:

			# 	print "Terminated after %d boosting steps since estimator_error = 0 \n" %(iboost+1)
			# 	break

			print "\n"

			b_tot_end = time.time()

			self.total_boosting_time.append(b_tot_end - b_tot_start)

			self.fitting_time.append(b_fit_end - b_fit_start)

			self.adv_generation_time.append(b_adv_gen_end - b_adv_gen_start)

			self.concatenate_time.append(b_adv_conc_end - b_adv_conc_start)

		return self



	def _boost(self, iboost, X, y):

			

		estimator = self._make_estimator()


		# estimator.fit(X, y, sample_weight=sample_weight)
		estimator.fit(X,y)

		self.estimators_.append(estimator)


		y_predict = estimator.predict(X)


		if iboost == 0:
			self.classes_ = getattr(estimator, 'classes_', None)
			self.n_classes_ = len(self.classes_)


		# Instances incorrectly classified
		incorrect = y_predict != y

		# Error fraction
		estimator_error = np.mean(np.average(incorrect, axis=0)) ####


		# Stop if classification is perfect
		# if estimator_error <= 0:

		# 	return estimator, 0                    ####

		n_classes = self.n_classes_


		# Stop if the error is at least as bad as random guessing
		# if estimator_error >= 1. - (1. / n_classes):

		# 	self.estimators_.pop(-1)
		# 	if len(self.estimators_) == 0:

		# 		raise ValueError('BaseClassifier in Adversarial Boost classifier '
		# 						 'ensemble is worse than random, ensemble '
		# 						 'can not be fit.')
		# 		return None, None                       ####

		
		return estimator, estimator_error


	def _make_estimator(self):

		estimator = clone(self.base_estimator)

		if(self.estimator_params is not None):

			estimator.set_params(**dict(self.estimator_params))

		return estimator


	def fitness_func(self, noise, x, clf):

		x = np.array(x).reshape(1,-1)
		return -1*norm(clf.predict_proba(x) - clf.predict_proba(x+noise))


	def _advGen(self, estimator, n_adv, n_boost_random_train_samples, epsilon, X, y):


		x0 = [0] * np.shape(X)[1]

		cons = ({'type': 'ineq', 'fun': lambda noise: epsilon - norm(np.array(noise))})

		optimal_noise = []
		optimal_fitness = []

		rand_idx = np.random.randint(0,len(X), (n_boost_random_train_samples,))
		rand_X = X[rand_idx]
		rand_y = y[rand_idx]

		for image_no, image in enumerate(rand_X):
			
			per_image_advGen_start = time.time()

			print "Generating adversarial examples for image number : %d\n" %image_no ,

			minimizer_kwargs = dict(method = "slsqp", args = (image, estimator), constraints = cons, options = {'maxiter': 100})
			res = basinhopping(self.fitness_func, niter = 10, x0 = x0, minimizer_kwargs = minimizer_kwargs)
			optimal_fitness.append(res['fun'])
			cur_noise = res['x']

			per_image_advGen_end = time.time()

			self.per_image_advGen_time.append(per_image_advGen_end - per_image_advGen_start)
			
			optimal_noise.append(cur_noise)

		index_noise = sorted(range(len(optimal_fitness)), key=lambda k: optimal_fitness[k])

		adv_examples = []
		adv_y = []
		for index in index_noise[:n_adv]:
			adv_examples.append(rand_X[index, :] + optimal_noise[index])
			adv_y.append(rand_y[index])

		return np.array(adv_examples), adv_y

no_boosting_clf = 10
no_adv_images = 100
epsilon = 0.3
no_base_estimator_trees = 50
no_basinhopping_iter = 10


with open('ab_10.pkl', 'rb') as f:
    data = pickle.load(f)

print('Hyperparameters: ')
print('Number of boosting classifier : {}'.format(no_boosting_clf))
print('Number of adversarial images : {}'.format(no_adv_images))
print('Epsilon value : {}'.format(epsilon))
print('Number of basinhopping iterations: {}'.format(no_basinhopping_iter))
print('Base Estimator: ')
print('Number of trees in base estimator: {}'.format(no_base_estimator_trees))
print('Splitting criteria : entropy')
print('Maximum depth : 5')
print('min samples split : 5')


print('\n\n Results::')

print('Average total boosting time: {}'.format(np.mean(data.total_boosting_time)))
print('Average training time : {}'.format(np.mean(data.fitting_time)))
print('Average total adversarial image generation time : {}'.format(np.mean(data.adv_generation_time)))
print('Average concatenation time :{}'.format(np.mean(data.concatenate_time)))
print('Average per adversarial image generation time of one image : {}'.format(np.mean(data.per_image_advGen_time)))