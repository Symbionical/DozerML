from copyreg import pickle
import pickle
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn import svm

participant_n = 19
channel_n = 16
# 0     1     2     3     4     5    6   7     8    9    10   11   12   13   14   15
# 'Fp1','Fp2','Fc5','Fz','Fc6','T7','Cz','T8','P7','P3','Pz','P4','P8','O1','Oz','O2',
#DOZER HALO ELECTRODES: T7,T8,P7,P8,O1,Oz,O2: [5 7 8 12 13 14 15]
dozer_halo = [5,7,8,12,13,14,15]
# we use fatigue ratings as our labels. repeat each element 10 times for the 10 trials.
# fatigue_ratings = [5, 0, 2, 2, 4, 7, 1, 7, 2, 0, 2, 0, 5, 4, 1, 4, 1, 3, 6, 2]. >= 4 is 1 "sleepy"
# fatigue_ratings = [1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0]
fatigue_ratings =   [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1] # why is 20 not there? 

y = np.repeat(fatigue_ratings[0:participant_n], 20) #10 = n trials
dfa = pd.read_pickle("bandpower_alpha.pkl")
Xna = dfa.to_numpy()
Xa = Xna[:, dozer_halo]
dft = pd.read_pickle("bandpower_theta.pkl")
Xnt = dft.to_numpy()
Xt = Xnt[:, dozer_halo]
X = Xa/Xt
# X = np.concatenate((Xna, Xnt), axis=1)
print(X.shape)
weights = {0:1.0, 1:4.0}

# Split data into a traning set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.10)
print("Fitting data to model... this may take a while")
classifier = svm.SVC(verbose=True, class_weight= weights).fit(X_train, y_train)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
print("Plotting data to confusion matrix")
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(classifier, X_test, y_test, cmap=plt.cm.Blues, normalize=normalize)
    # disp.ax.set_title(title)
    print(title)
    print(disp.confusion_matrix)

plt.show()
print("Making final fit")
classifier = svm.SVC(verbose=True, class_weight= weights).fit(X, y)
print("Saving classifier to disk")
s = pickle.dump(classifier, open('sleepy_class.sav', 'wb'))