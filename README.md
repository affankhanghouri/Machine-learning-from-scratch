# Machine Learning from Scratch 

Welcome to my personal journey of understanding how machine learning really works **under the hood**.  
This repo isn't meant to be super polished or production-ready it's just a **learning playground** where I build everything from scratch using only **NumPy**.

---

##  What's the goal?

I got tired of just calling `sklearn.fit()` and moving on.

So I decided to recreate the entire ML workflow by hand from scratch  so I could understand:
- How algorithms actually learn
- What happens during training
- Why certain functions and modules even exist

---

##  What I’ve Built So Far

All done using **NumPy** and my own utility code  no `sklearn` models used.

###  Custom ML Algorithms
- ✅ Linear Regression (from scratch)
- ✅ Binary Logistic Regression
- ✅ Multiclass Logistic Regression (One-vs-Rest)
- ✅ Decision Tree Classifier
- ✅ Random Forest Classifier

###  Custom Tools & Utilities
- `train_test_split()` — my own version
- `LabelEncoder` — simple implementation
- `StandardScaler` — built from scratch
- Custom Exception Handling —> to practice writing cleaner, modular code
- Everything modularized into clean Python classes & functions

---

##  Why This Matters to Me

I’m doing this to:
- Get better at **core math** (loss functions, gradients, entropy, etc.)
- Improve my **Python and OOP design**
- Understand **how machine learning libraries actually work**
- Train myself to think like an engineer, not just a user

---

## Example Results

Here’s an example:  
With my custom **Random Forest Classifier** (pure NumPy), I achieved **~95% accuracy** on the Breast Cancer dataset.  
And I didn’t even use `sklearn.ensemble`  every tree, every split, every vote was done manually.

---

##  What's Next?

This is ongoing. As I go deeper, I plan to add:
- Naive Bayes
- KNN
- K-Means
- PCA
- Backpropagation from scratch
- Maybe even build a tiny framework

Let’s see how far this playground can go :)

---

