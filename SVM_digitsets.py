
import numpy as np
from SMO_notes import SMO
np.random.seed(42)
import ray
from  sklearn.metrics import accuracy_score
ray.init()



class SVM_Digits:
    def read_data(self):
        X = np.loadtxt("./features20.txt")
        Y = np.loadtxt("./labels20.txt")
        data = np.c_[X, Y]
        np.random.shuffle(data)
        train_size = int(data.shape[0] * 0.8)
        X_train = data[:train_size, :-1]
        X_test = data[train_size:, :-1]
        Y_train = data[:train_size, -1]
        Y_test = data[train_size:, -1]
        return X_train, Y_train, X_test, Y_test

    def get_new_labels(self, Y_train, Y_test, i):
        temp_Y_train  = np.where(Y_train != i, -1, 1)
        temp_Y_test = np.where(Y_test != i, -1, 1)
        return temp_Y_train, temp_Y_test

    @ray.remote
    def parallel(self, X_train, Y_train, X_test, Y_test, i, f):
        temp_Y_train, temp_Y_test = self.get_new_labels(Y_train, Y_test, i)
        svm_smo = SMO(C=0.01, tol=0.01, max_passes=100, epochs=100, f=f)
        svm_smo.fit(X_train, temp_Y_train)
        y_pred_train = svm_smo.predict_scores(X_train).reshape(Y_train.shape[0], 1)
        y_pred_test = svm_smo.predict_scores(X_test).reshape(Y_test.shape[0], 1)
        return y_pred_train, y_pred_test

    def append(self, predictions, y_pred0, y_pred1, y_pred2, y_pred3, y_pred4):
        predictions = np.c_[predictions, y_pred0, y_pred1, y_pred2, y_pred3, y_pred4]
        return predictions

    def run(self):
        X_train, Y_train, X_test, Y_test = self.read_data()
        print("done reading")
        train_predictions = np.zeros(Y_train.shape[0]).reshape(Y_train.shape[0], 1)
        test_predictions = np.zeros(Y_test.shape[0]).reshape(Y_test.shape[0], 1)
        with open("SVM_digitsets.txt", "w") as f:
            for i in range(10):
                print("digit = ", i)
                Id1 = self.parallel.remote(X_train, Y_train, X_test, Y_test, i)
                Id2 = self.parallel.remote(X_train, Y_train, X_test, Y_test, i + 1)
                Id3 = self.parallel.remote(X_train, Y_train, X_test, Y_test, i + 2)
                Id4 = self.parallel.remote(X_train, Y_train, X_test, Y_test, i + 3)
                Id5 = self.parallel.remote(X_train, Y_train, X_test, Y_test, i + 4)
                y_pred_train0, y_pred_test0 = ray.get(Id1)
                y_pred_train1, y_pred_test1 = ray.get(Id2)
                y_pred_train2, y_pred_test2 = ray.get(Id3)
                y_pred_train3, y_pred_test3 = ray.get(Id4)
                y_pred_train4, y_pred_test4 = ray.get(Id5)

                train_predictions = self.append(train_predictions, y_pred_train0, y_pred_train1, y_pred_train2, y_pred_train3, y_pred_train4)
                test_predictions = self.append(test_predictions, y_pred_test0, y_pred_test1, y_pred_test2, y_pred_test3, y_pred_test4)
            multiclass_train_prediction = np.argmax(train_predictions, axis=1)
            multiclass_test_prediction = np.argmax(test_predictions, axis = 1)
            print("Testing accuracy = ", accuracy_score(Y_test, multiclass_test_prediction))
            print("Training accuracy = ",  accuracy_score(Y_train, multiclass_train_prediction))


# X, Y = read_mnist()
# haar = HaarFeatures(100)
# features = haar.getFeatures(X)
# np.savetxt("./features20.txt", features)
# np.savetxt("./labels20.txt", Y)

digits = SVM_Digits()
digits.run()






