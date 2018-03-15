from utils import *
from tempfile import TemporaryFile
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
X_train, y_train, X_test, y_test = extract(mnist)
print(X_train.shape)
clf = createRandomForest(X_train, y_train)

#X_train_adv = adversarialize(X_train, clf)

X_train_adv = adversarialize(X_train, clf)
np.savez("adv_train.npz", X_train_adv)
# test_file = TemporaryFile()
# test_file.savez(test_file, X_test=X_test, y_test=y_test)
# adv_test_file = TemporaryFile()
# adv_test_file.savez(adv_test_file, X_test_adv=X_test_adv, y_test=y_test)