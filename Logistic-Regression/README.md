# Binary Logistic Regression (From Scratch)

Simple, clean, and fully manual binary logistic regression using just NumPy.  
No `sklearn`, no magic — just raw math and learning.

---

##  What i did

- Created a `My_custom_logisticRegression` class
- Used **sigmoid activation** and **binary cross-entropy loss**
- Built a full training loop with **manual gradient descent**
- Tracked loss over epochs for visualization
- Option to **plot the loss curve**
- used custom exception , custom preprocessing utilis like (train_test_split , standard scaler etc)

---

##  How It Works

- `fit(X, y)` trains the model from scratch
- `predict_proba(X)` gives raw probabilities
- `predict(X)` gives final class labels (0 or 1)
- `plot_loss()` shows how loss decreased over training

---

## Goal

To truly understand how logistic regression trains  not just use it as a black box.

