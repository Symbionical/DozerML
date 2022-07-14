import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

participant_n = 19
channel_n = 16
# 0     1     2     3     4     5    6   7     8    9    10   11   12   13   14   15
# 'Fp1','Fp2','Fc5','Fz','Fc6','T7','Cz','T8','P7','P3','Pz','P4','P8','O1','Oz','O2',
#DOZER HALO ELECTRODES: T7,T8,P7,P8,O1,Oz,O2: [5 7 8 12 13 14 15]
dozer_halo = [5,7,8,12,13,14,15]
# we use fatigue ratings as our labels. repeat each element 10 times for the 10 trials.
# fatigue_ratings = [5, 0, 2, 2, 4, 7, 1, 7, 2, 0, 2, 0, 5, 4, 1, 4, 1, 3, 6, 2]. >= 5 is 1 "sleepy"
# fatigue_ratings = [1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0]
fatigue_ratings = [1,0,0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,0,1] # why is 20 not there? 

y = np.repeat(fatigue_ratings[0:participant_n], 10) #10 = n trials
df = pd.read_pickle("bandpower_alpha.pkl")
Xn = df.to_numpy()
X = Xn[:, dozer_halo]

skf = StratifiedKFold(n_splits=5)
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
scr = cross_val_score(clf, X, y, cv=skf)

# print results of classification
print('mean accuracy :', scr.mean())