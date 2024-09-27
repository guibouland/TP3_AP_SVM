#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

from svm_source import *
from sklearn import svm
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from time import time

scaler = StandardScaler()

import warnings
warnings.filterwarnings("ignore")

plt.style.use('ggplot')


#%%
###############################################################################
#               Toy dataset : 2 gaussians
###############################################################################

n1 = 200
n2 = 200
mu1 = [1., 1.]
mu2 = [-1./2, -1./2]
sigma1 = [0.9, 0.9]
sigma2 = [0.9, 0.9]
X1, y1 = rand_bi_gauss(n1, n2, mu1, mu2, sigma1, sigma2)

plt.show()
plt.close("all")
plt.ion()
plt.figure(1, figsize=(15, 5))
plt.title('First data set')
plot_2d(X1, y1)

X_train = X1[::2]
Y_train = y1[::2].astype(int)
X_test = X1[1::2]
Y_test = y1[1::2].astype(int)

# fit the model with linear kernel
clf = SVC(kernel='linear')
clf.fit(X_train, Y_train)

# predict labels for the test data base
y_pred = clf.predict(X_test)

# check your score
score = clf.score(X_test, Y_test)
print('Score : %s' % score)

# display the frontiere
def f(xx):
    """Classifier: needed to avoid warning due to shape issues"""
    return clf.predict(xx.reshape(1, -1))

plt.figure()
frontiere(f, X_train, Y_train, w=None, step=50, alpha_choice=1)

# Same procedure but with a grid search
parameters = {'kernel': ['linear'], 'C': list(np.linspace(0.001, 3, 21))}
clf2 = SVC()
clf_grid = GridSearchCV(clf2, parameters, n_jobs=-1)
clf_grid.fit(X_train, Y_train)

# check your score
print(clf_grid.best_params_)
print('Score : %s' % clf_grid.score(X_test, Y_test))

def f_grid(xx):
    """Classifier: needed to avoid warning due to shape issues"""
    return clf_grid.predict(xx.reshape(1, -1))

# display the frontiere
plt.figure()
frontiere(f_grid, X_train, Y_train, w=None, step=50, alpha_choice=1)

#%%
###############################################################################
#               Iris Dataset
###############################################################################

iris = datasets.load_iris()
X = iris.data
X = scaler.fit_transform(X)
y = iris.target
X = X[y != 0, :2]
y = y[y != 0]

# display dataset
plt.figure(1, figsize=(15, 5))
plt.title('Iris dataset')
plot_2d(X, y)

#%%
###############################################################################
# fit the model with linear vs polynomial kernel
###############################################################################
# split train test
X, y = shuffle(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Q1 Linear kernel

# fit the model with CV
parameters = {'kernel': ['linear'], 'C': list(np.logspace(-3, 3, 200))}
clf_linear = GridSearchCV(SVC(), parameters, n_jobs=-1)
clf_linear.fit(X_train, y_train)
# compute the scores
print('Generalization score for linear kernel: %s, %s' %
      (clf_linear.score(X_train, y_train),
       clf_linear.score(X_test, y_test)))

#plot the frontiere
def f_linear(xx):
    """Classifier: needed to avoid warning due to shape issues"""
    return clf_linear.predict(xx.reshape(1, -1))

plt.figure()
frontiere(f_linear, X, y)
plt.title("linear kernel with CV")
plt.show()
#%%
# Whitout CV
clf_linear2 = SVC(kernel='linear')
clf_linear2.fit(X_train, y_train)
print('Generalization score for linear kernel: %s, %s' %
      (clf_linear2.score(X_train, y_train),
       clf_linear2.score(X_test, y_test)))

#plot the frontiere
def f_linear2(xx):
    """Classifier: needed to avoid warning due to shape issues"""
    return clf_linear.predict(xx.reshape(1, -1))
plt.figure()
frontiere(f_linear2, X, y)
plt.title("linear kernel without CV")
plt.show()
#%%
# faire une myenne des scores pour avoir un score plus stable
score_train = 0
score_test = 0
for i in range(30):
    clf_linear = GridSearchCV(SVC(), parameters, n_jobs=-1)
    clf_linear.fit(X_train, y_train)
    score_train += clf_linear.score(X_train, y_train)
    score_test += clf_linear.score(X_test, y_test)
score_train /= 30
score_test /= 30

print("Score moyen sur 30 iterations", score_train, score_test)
#%%
# moyenne sans CV
score_train = 0
score_test = 0
for i in range(30):
    clf_linear = SVC(kernel='linear')
    clf_linear.fit(X_train, y_train)
    score_train += clf_linear.score(X_train, y_train)
    score_test += clf_linear.score(X_test, y_test)
score_train /= 30
score_test /= 30

print("Score moyen sur 30 iterations", score_train, score_test)
#%%
# Q2 polynomial kernel with CV
Cs = list(np.logspace(-3, 3, 5))
gammas = 10. ** np.arange(1, 2)
degrees = np.r_[1, 2, 3]

parameters = {'kernel': ['poly'], 'C': Cs, 'gamma': gammas, 'degree': degrees}
clf_poly = GridSearchCV(SVC(), parameters, n_jobs=-1)
clf_poly.fit(X_train, y_train)
#%%
print(clf_poly.best_params_)
# compute the scores
print('Generalization score for polynomial kernel: %s, %s' %
    (clf_poly.score(X_train, y_train),
    clf_poly.score(X_test, y_test)))

#plot the frontiere
def f_poly(xx):
    """Classifier: needed to avoid warning due to shape issues"""
    return clf_poly.predict(xx.reshape(1, -1))

plt.figure()
frontiere(f_poly, X, y)
plt.title("polynomial kernel with CV")
plt.show()

#%%
# faire une myenne des scores pour avoir un score plus stable
score_train = 0
score_test = 0
for i in range(10):
    clf_poly = GridSearchCV(SVC(), parameters, n_jobs=-1)
    clf_poly.fit(X_train, y_train)
    score_train += clf_poly.score(X_train, y_train)
    score_test += clf_poly.score(X_test, y_test)
score_train /= 10
score_test /= 10
print("Score moyen sur 10 iterations", score_train, score_test)

#%%
# polynomial kernel without CV
iris = datasets.load_iris()
X = iris.data
X = scaler.fit_transform(X)
y = iris.target
X = X[y != 0, :2]
y = y[y != 0]
X, y = shuffle(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

clf_poly2 = SVC(kernel='poly')
clf_poly2.fit(X_train, y_train)
print('Generalization score for polynomial kernel: %s, %s' %
    (clf_poly.score(X_train, y_train),
    clf_poly.score(X_test, y_test)))

def f_poly2(xx):
    """Classifier: needed to avoid warning due to shape issues"""
    return clf_poly2.predict(xx.reshape(1, -1))
# plot the frontiere
plt.figure()
frontiere(f_poly2, X_train, y_train, w=None, step=50, alpha_choice=1)
plt.title("polynomial kernel without CV")
plt.show()

#%%
# faire une myenne des scores pour avoir un score plus stable
score_train = 0
score_test = 0
for i in range(30):
    clf_poly = SVC(kernel='poly')
    clf_poly.fit(X_train, y_train)
    score_train += clf_poly.score(X_train, y_train)
    score_test += clf_poly.score(X_test, y_test)
score_train /= 30
score_test /= 30
print("Score moyen sur 30 iterations", score_train, score_test)

#%%
# display your results using frontiere

def f_linear(xx):
    """Classifier: needed to avoid warning due to shape issues"""
    return clf_linear.predict(xx.reshape(1, -1))

def f_poly(xx):
    """Classifier: needed to avoid warning due to shape issues"""
    return clf_poly.predict(xx.reshape(1, -1))

plt.ion()
plt.figure(figsize=(15, 5))
plt.subplot(131)
plot_2d(X, y)
plt.title("iris dataset")

plt.subplot(132)
frontiere(f_linear, X, y)
plt.title("linear kernel")

plt.subplot(133)
frontiere(f_poly, X, y)

plt.title("polynomial kernel")
plt.tight_layout()
plt.draw()

#%%
# for the polynomial kernel
plt.ion()
plt.figure(figsize=(15, 5))
plt.subplot(131)
plot_2d(X, y)
plt.title("iris dataset")

plt.subplot(132)
frontiere(f_linear2, X, y)
plt.title("linear kernel")

plt.subplot(133)
frontiere(f_poly2, X, y)
plt.title("polynomial kernel")
plt.tight_layout()
plt.draw()

#%%
###############################################################################
#               SVM GUI
###############################################################################

# please open a terminal and run python svm_gui.py
# Then, play with the applet : generate various datasets and observe the
# different classifiers you can obtain by varying the kernel


#%%
###############################################################################
#               Face Recognition Task
###############################################################################
"""
The dataset used in this example is a preprocessed excerpt
of the "Labeled Faces in the Wild", aka LFW_:

  http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)

  _LFW: http://vis-www.cs.umass.edu/lfw/
"""

####################################################################
# Download the data and unzip; then load it as numpy arrays
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4,
                              color=True, funneled=False, slice_=None,
                              download_if_missing=True)
# data_home='.'

# introspect the images arrays to find the shapes (for plotting)
images = lfw_people.images
n_samples, h, w, n_colors = images.shape

# the label to predict is the id of the person
target_names = lfw_people.target_names.tolist()

####################################################################
# Pick a pair to classify such as
names = ['Tony Blair', 'Colin Powell']
# names = ['Donald Rumsfeld', 'Colin Powell']

idx0 = (lfw_people.target == target_names.index(names[0]))
idx1 = (lfw_people.target == target_names.index(names[1]))
images = np.r_[images[idx0], images[idx1]]
n_samples = images.shape[0]
y = np.r_[np.zeros(np.sum(idx0)), np.ones(np.sum(idx1))].astype(int)

# plot a sample set of the data
plot_gallery(images, np.arange(12))
plt.show()

#%%
####################################################################
# Extract features

# features using only illuminations
X = (np.mean(images, axis=3)).reshape(n_samples, -1)

# # or compute features using colors (3 times more features)
# X = images.copy().reshape(n_samples, -1)

# Scale features
X -= np.mean(X, axis=0)
X /= np.std(X, axis=0)

#%%
####################################################################
# Split data into a half training and half test set
# X_train, X_test, y_train, y_test, images_train, images_test = \
#    train_test_split(X, y, images, test_size=0.5, random_state=0)
# X_train, X_test, y_train, y_test = \
#    train_test_split(X, y, test_size=0.5, random_state=0)

indices = np.random.permutation(X.shape[0])
train_idx, test_idx = indices[:X.shape[0] // 2], indices[X.shape[0] // 2:]
X_train, X_test = X[train_idx, :], X[test_idx, :]
y_train, y_test = y[train_idx], y[test_idx]
images_train, images_test = images[
    train_idx, :, :, :], images[test_idx, :, :, :]

####################################################################
# Quantitative evaluation of the model quality on the test set

#%%
# Q3
print("--- Linear kernel ---")
print("Fitting the classifier to the training set")
t0 = time()

# fit a classifier (linear) and test all the Cs
Cs = 10. ** np.arange(-5, 6)
scores = []
errors = []
for C in Cs:
    clf = svm.SVC(kernel='linear', C=C)
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_train, y_train))
    errors.append(1 - clf.score(X_train, y_train))
    print("C: {}, score: {}".format(C, scores[-1]))
    print("C: {}, error: {}".format(C, errors[-1]))
ind = np.argmax(scores)
ind2 = np.argmin(errors)
print("Best C: {}".format(Cs[ind]))
print("Best C: {}".format(Cs[ind2]))

#%%
plt.ion()
plt.figure(figsize=(15, 5))
plt.subplot(121)
plt.plot(Cs, scores)
plt.title("Scores d'apprentissage")
plt.xlabel("Parametres de regularisation C")
plt.ylabel("Scores d'apprentissage")
plt.xscale("log")


plt.subplot(122)
plt.plot(Cs, errors)
plt.title("Erreurs d'apprentissage")
plt.xlabel("Parametres de regularisation C")
plt.ylabel("Erreurs d'apprentissage")
plt.xscale("log")

plt.tight_layout()
plt.draw()
print("Best score: {}".format(np.max(scores)))
print("Best C: {}".format(Cs[ind]))
print("Predicting the people names on the testing set")
t0 = time()





#%%
# predict labels for the X_test images with the best classifier

clf = svm.SVC(kernel='linear', C=Cs[ind])
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


print("done in %0.3fs" % (time() - t0))
# The chance level is the accuracy that will be reached when constantly predicting the majority class.
print("Chance level : %s" % max(np.mean(y_pred), 1. - np.mean(y_pred)))
print("Accuracy : %s" % clf.score(X_test, y_test))



#%%
####################################################################
# Qualitative evaluation of the predictions using matplotlib

prediction_titles = [title(y_pred[i], y_test[i], names)
                     for i in range(y_pred.shape[0])]

plot_gallery(images_test, prediction_titles)
plt.show()

#%%
####################################################################
# Look at the coefficients
plt.figure()
plt.imshow(np.reshape(clf.coef_, (h, w)))
plt.show()


#%%
# Q4

def run_svm_cv(_X, _y):
    _indices = np.random.permutation(_X.shape[0])
    _train_idx, _test_idx = _indices[:_X.shape[0] // 2], _indices[_X.shape[0] // 2:]
    _X_train, _X_test = _X[_train_idx, :], _X[_test_idx, :]
    _y_train, _y_test = _y[_train_idx], _y[_test_idx]

    _parameters = {'kernel': ['linear'], 'C': list(np.logspace(-3, 3, 5))}
    _svr = svm.SVC()
    _clf_linear = GridSearchCV(_svr, _parameters)
    _clf_linear.fit(_X_train, _y_train)

    print('Generalization score for linear kernel: %s, %s \n' %
          (_clf_linear.score(_X_train, _y_train), _clf_linear.score(_X_test, _y_test)))
    return _clf_linear.score(_X_test, _y_test)
print("Score sans variable de nuisance")
run_svm_cv(X, y)
#%%
sc=[]
for i in range(10):
    sc.append(run_svm_cv(X, y))
print("Score moyen sur 10 iterations", np.mean(sc))
run_svm_cv(X, y)


#%%


print("Score avec variable de nuisance")
n_features = X.shape[1]
# On rajoute des variables de nuisances
sigma = 1
noise = sigma * np.random.randn(n_samples, 300, )
X_noisy = np.concatenate((X, noise), axis=1)
X_noisy = X_noisy[np.random.permutation(X.shape[0])]
print(X_noisy.shape)

run_svm_cv(X_noisy, y)

#%%
sc2 = []
for i in range(10):
    n_features = X.shape[1]
    # On rajoute des variables de nuisances
    sigma = 1
    noise = sigma * np.random.randn(n_samples, 300, )
    X_noisy = np.concatenate((X, noise), axis=1)
    X_noisy = X_noisy[np.random.permutation(X.shape[0])]
    sc2.append(run_svm_cv(X_noisy, y))
print("Score moyen sur 10 iterations", np.mean(sc2))


#%%
# Q5

print("Score apres reduction de dimension")

n_components = 120  # jouer avec ce parametre
pca = PCA(n_components=n_components).fit(X_noisy)

X_pca = pca.transform(X_noisy)
run_svm_cv(X_pca, y)


#%%

pca2 = PCA(n_components=60, svd_solver='randomized').fit(X_noisy)
#%%
X_pca2 = pca2.transform(X_noisy)
print(X_pca2.shape)

run_svm_cv(X_pca2, y)
#%%
# BOUCLE
sc3 = []
for i in range(10):
    pca = PCA(n_components=80, svd_solver='randomized').fit(X_noisy)
    X_pca = pca.transform(X_noisy)
    sc3.append(run_svm_cv(X_pca, y))
print("Score moyen sur 10 iterations", np.mean(sc3))
#%%
