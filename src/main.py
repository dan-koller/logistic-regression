import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class CustomLogisticRegression:

    def __init__(self, fit_intercept=True, l_rate=0.01, n_epoch=100):
        # Initialize the model
        self.fit_intercept = fit_intercept
        self.l_rate = l_rate
        self.n_epoch = n_epoch
        # Additional attributes for the model to compare with sklearn
        self.first_mse = []
        self.last_mse = []
        self.first_log_loss = []
        self.last_log_loss = []

    def sigmoid(self, t):
        # Sigmoid function
        return 1 / (1 + np.exp(-t))

    def predict_proba(self, row, coef_):
        # Predict the probability of a data point belonging to a class
        if self.fit_intercept:
            row = np.insert(row, 0, np.ones(row.shape[0]), axis=1)
        t = np.dot(row, coef_)
        return self.sigmoid(t)

    def mse(self, X, y):
        # Mean squared error function
        y_hat = self.predict_proba(X, self.coef_)
        return np.mean((y - y_hat) ** 2)

    def update_mse(self, row, y_train):
        # Method to update the mse for a single data point
        y_hat = self.predict_proba(row, self.coef_)
        gradient = (y_hat - y_train) * y_hat * (1 - y_hat)
        self.coef_ += -self.l_rate * gradient * row

    def fit_mse(self, X_train, y_train):
        n = X_train.shape[0]
        if self.fit_intercept:
            X_train = np.insert(X_train, 0, np.ones(n), axis=1)
        self.coef_ = np.zeros(X_train.shape[1])
        self.fit_intercept = False

        # Append the first mse
        for i in range(n):
            self.update_mse(X_train[i], y_train[i])
            self.first_mse.append(self.mse(X_train, y_train))

        # Calculate the mse for every epoch
        for _ in range(self.n_epoch):
            for i in range(n):
                self.update_mse(X_train[i], y_train[i])

        # Append the last mse
        for i in range(n):
            self.update_mse(X_train[i], y_train[i])
            self.last_mse.append(self.mse(X_train, y_train))

        self.fit_intercept = True

    def log_loss(self, X, y):
        # Log loss function
        y_hat = self.predict_proba(X, self.coef_)
        return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

    def update_log_loss(self, row, y_train, n):
        # Method to update the log loss for a single data point
        y_hat = self.predict_proba(row, self.coef_)
        error = y_hat - y_train
        gradient = error * row / n
        self.coef_ += -self.l_rate * gradient

    def fit_log_loss(self, X_train, y_train):
        # Stochastic gradient descent implementation
        n = X_train.shape[0]
        if self.fit_intercept:
            X_train = np.insert(X_train, 0, np.ones(n), axis=1)
        self.coef_ = np.zeros(X_train.shape[1])
        self.fit_intercept = False

        # Append the first log loss
        for i in range(n):
            self.update_log_loss(X_train[i], y_train[i], n)
            self.first_log_loss.append(self.log_loss(X_train, y_train))

        # Calculate the log loss for every epoch
        for _ in range(self.n_epoch):
            for i in range(n):
                self.update_log_loss(X_train[i], y_train[i], n)

        # Append the last log loss
        for i in range(n):
            self.update_log_loss(X_train[i], y_train[i], n)
            self.last_log_loss.append(self.log_loss(X_train, y_train))

        self.fit_intercept = True

    def predict(self, X_test, cut_off=0.5):
        # Predict the class labels for the test data
        y_hat = self.predict_proba(X_test, self.coef_)
        predictions = np.where(y_hat >= cut_off, 1, 0)
        return predictions  # predictions are binary values - 0 or 1


def load_data():
    # Load the breast cancer dataset from sklearn
    data = load_breast_cancer()
    col_names = ['worst concave points', 'worst perimeter', 'worst radius']
    X = data.data[:, [data['feature_names'].tolist().index(name)
                      for name in col_names]]
    y = data.target
    return X, y


def remove_empty_lines(string):
    # Removes empty lines from the string & returns a list split by \n separator
    string_splitted = string.split("\n")
    string_splitted_with_no_empty_lines = [
        line for line in string_splitted if line.strip() != ""]
    return string_splitted_with_no_empty_lines


def plot(result_dict):
    # Plot the results
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
    axs[0, 0].plot(result_dict['mse_error_first'])
    axs[0, 1].plot(result_dict['mse_error_last'])
    axs[1, 0].plot(result_dict['logloss_error_first'])
    axs[1, 1].plot(result_dict['logloss_error_last'])
    axs[0, 0].set_ylabel('Squared Error')
    axs[0, 0].set_title('MSE: First Epoch Errors')
    axs[0, 1].set_ylabel('Squared Error')
    axs[0, 1].set_title('MSE: Last Epoch Errors')
    axs[1, 0].set_ylabel('Log Loss')
    axs[1, 0].set_title('Log-loss: First Epoch Errors')
    axs[1, 1].set_ylabel('Log Loss')
    axs[1, 1].set_title('Log-loss: Last Epoch Errors')
    # plt.show()
    plt.savefig("data/graph.jpg")


# Load data, standardize it and split into train and test sets
X, y = load_data()
X = stats.zscore(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=43)

# Create a new model and fit it
model = CustomLogisticRegression(fit_intercept=True, l_rate=0.01, n_epoch=1000)

# Make predictions about the mse and log loss
model.fit_mse(X_train, y_train)
y_pred_mse = model.predict(X_test)

model.fit_log_loss(X_train, y_train)
y_pred_log_loss = model.predict(X_test)

# Create a new sklearn model (to compare the custom model with)
skmodel = LogisticRegression(fit_intercept=True, solver='lbfgs', max_iter=1000)
skmodel.fit(X_train, y_train)
y_pred_sk = skmodel.predict(X_test)

# Evaluate the model with the metrics
res = {'mse_accuracy': accuracy_score(y_pred_mse, y_test),
       'logloss_accuracy': accuracy_score(y_pred_log_loss, y_test),
       'sklearn_accuracy': accuracy_score(y_pred_sk, y_test),
       'mse_error_first': model.first_mse,
       'mse_error_last': model.last_mse,
       'logloss_error_first': model.first_log_loss,
       'logloss_error_last': model.last_log_loss}
print(res)

# Plot the results
plot(res)
