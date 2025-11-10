# ----------------------------------------------
# SVM (Polynomial Kernel) From Scratch
# Breast Cancer Classification
# ----------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert labels to -1 and +1 for SVM
y = np.where(y == 0, -1, 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Polynomial Kernel Function
def polynomial_kernel(x1, x2, degree=3, coef0=1):
    return (np.dot(x1, x2.T) + coef0) ** degree


# SVM Class
class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y[idx]))
                    self.b -= self.lr * y[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)


# Apply Polynomial Kernel transformation
def polynomial_features(X, degree=3):
    poly = []
    for i in range(X.shape[0]):
        poly.append(np.power(X[i], degree))
    return np.array(poly)


# Transform data
X_train_poly = polynomial_features(X_train, degree=2)
X_test_poly = polynomial_features(X_test, degree=2)

# Train SVM
svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=2000)
svm.fit(X_train_poly, y_train)

# Predict
y_pred = svm.predict(X_test_poly)

# Convert back to 0/1
y_pred_binary = np.where(y_pred == 1, 1, 0)
y_true_binary = np.where(y_test == 1, 1, 0)

# Evaluate
cm = confusion_matrix(y_true_binary, y_pred_binary)
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_true_binary, y_pred_binary))

# ROC Curve
fpr, tpr, _ = roc_curve(y_true_binary, y_pred_binary)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label='SVM (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — SVM Polynomial Kernel')
plt.legend(loc='lower right')
plt.show()
