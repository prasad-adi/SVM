import pandas as pd
import numpy as np
import sklearn
np.random.seed(0)
from sklearn.metrics import accuracy_score



class SMO:
    def __init__(self, C, tol, max_passes, epochs, f):
        self.C = C
        self.tol = tol
        self.max_passes = max_passes
        self.epochs = epochs
        self.f = f

    def __calculate_F_X(self, x):
        return (self.alphas * self.Y * (np.multiply(self.X, x).sum(axis = 1))).sum() + self.b

    def __get_E(self, x, y):
        F = self.__calculate_F_X(x)
        return F - y

    def __get_alpha_j(self, H, L, alpha_j):
        if(alpha_j > H):
            return H
        elif(L <= alpha_j <= H):
            return alpha_j
        elif(alpha_j < L):
            return L

    def __get_b(self, b1,b2, alphas,i,j):
        if(0 < alphas[i] < self.C):
            return b1
        elif(0 < alphas[j] < self.C):
            return b2
        else:
            return (b1 + b2) / 2


    def fit(self, X, Y):
        self.X = X
        self.alphas = np.zeros(X.shape[0])
        self.b = 0
        self.Y = Y
        passes = 0
        for epoch in range(self.epochs):
            num_changed_alphas = 0
            for i in range(self.X.shape[0]):
                E_i = self.__get_E(self.X[i], self.Y[i])
                if ((Y[i] * E_i < -self.tol and self.alphas[i] < self.C) or (self.Y[i] * E_i > self.tol and self.alphas[i] > 0)):
                    j = i
                    while(j == i):
                        j = np.random.randint(0, self.X.shape[0], size=1)[0]
                    E_j = self.__get_E(self.X[j], self.Y[j])
                    alpha_i_old = self.alphas[i]
                    alpha_j_old = self.alphas[j]
                    if(self.Y[i] != self.Y[j]):
                        L = max(0, self.alphas[j] - self.alphas[i])
                        H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
                    else:
                        L = max(0, self.alphas[i] + self.alphas[j] - self.C)
                        H = min(self.C, self.alphas[i] + self.alphas[j])
                    if(L == H):
                        continue
                    eta = 2 * np.dot(self.X[i], self.X[j]) - np.dot(self.X[i], self.X[i]) - np.dot(self.X[j], self.X[j])
                    if (eta >= 0):
                        continue
                    alpha_j_temp = self.alphas[j] - ((self.Y[j] * (E_i - E_j)) / eta)
                    self.alphas[j] = self.__get_alpha_j(H, L, alpha_j_temp)
                    if(abs(self.alphas[j] - alpha_j_old) < 1e-5):
                        continue
                    self.alphas[i] = self.alphas[i] + self.Y[i] * self.Y[j] * (alpha_j_old - self.alphas[j])
                    b1 = self.b - E_i - (self.Y[i] * (self.alphas[i] - alpha_i_old) * np.dot(self.X[i], self.X[i])) - (self.Y[j] * (self.alphas[j] - alpha_j_old) * np.dot(self.X[i], self.X[j]))
                    b2 = self.b - E_j - (self.Y[i] * (self.alphas[i] - alpha_i_old) * np.dot(self.X[i], self.X[j])) - (self.Y[j] * (self.alphas[j] - alpha_j_old) * np.dot(self.X[j], self.X[j]))
                    self.b = self.__get_b(b1,b2, self.alphas,i,j)
                    num_changed_alphas += 1
            if(epoch % 10 == 0):
                y_pred = self.predict(self.X)
                self.f.write("\n" + str(epoch))
                self.f.write("\naccuracy = " + str(accuracy_score(self.Y, y_pred)))

                print(epoch)
                print("accuracy = ", accuracy_score(self.Y, y_pred))

            if(num_changed_alphas == 0):
                passes += 1
            else:
                passes = 0


    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            y_pred.append(self.__calculate_F_X(X[i]))
        y_pred = np.array(y_pred)
        y_pred = np.where(y_pred <= 0, -1, 1)
        return y_pred

    def predict_scores(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            y_pred.append(self.__calculate_F_X(X[i]))
        y_pred = np.array(y_pred)
        return y_pred





