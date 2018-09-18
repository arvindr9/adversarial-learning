#importing modules
from sklearn.ensemble import RandomForestClassifier
from scipy.optimize import minimize
from scipy.linalg import norm
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import basinhopping
import csv
from sklearn.metrics import accuracy_score
import joblib
from joblib import Parallel, delayed
# import cPickle
import argparse
from sklearn.externals import joblib

class adv_test:

    def init(self, config, clf, data, epsilon):
        self.base_classifier = config['base_classifier']
        self.epsilon = epsilon
        self.data = data
        self.n_estimators = clf
        self.inner_iter = config['inner_iter']
        self.outer_iter = config['outer_iter']
        self.no_threads = config['no_threads']
        if self.base_classifier == 'random_forest':
            self.data_path = config['data_path']
            self.max_depth = config['max_depth']
            self.criterion = config['criterion']
            self.n_estimators = config['n_estimators']
        self.main()
    
    def main():
        