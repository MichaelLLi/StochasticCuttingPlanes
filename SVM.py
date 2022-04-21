# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 12:34:49 2020

@author: Michael
"""

import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
import time
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("data/covtype.data",header=None)
n = 100000
X = df.iloc[:,list(range(54))]
scaler = StandardScaler()
y = (df.iloc[:,54] == 2).astype(int) * 2 - 1
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    train_size=1000 ,
                                                    test_size = 1000,
                                                    random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = SGDClassifier(loss="hinge", penalty="l2", shuffle=True, max_iter=1000000,alpha = 1 / 1e6, learning_rate = 'constant', eta0 = 0.05)
start = time.time()
clf.fit(X_train_scaled, y_train)
end = time.time()
print(end - start)
y_pred = clf.predict(X_test_scaled)
np.mean(y_pred == y_test)