# 🌳 Custom Random Forest Classifier (from Scratch using NumPy)

This project implements a **Random Forest Classifier from scratch using NumPy** only  no `sklearn`, `xgboost`, or other libraries. Just pure logic, decision trees, and bootstrapping!

---

##  What I Did

- Built a **Decision Tree Classifier** from scratch.
- Then extended it into a full **Random Forest** by:
  - Bootstrapping the dataset
  - Training multiple decision trees
  - Aggregating predictions with majority voting
- Designed everything to be **simple, readable**.

---

##  Dataset Used

Used the **Breast Cancer Wisconsin dataset** (binary classification):
- 0 = malignant
- 1 = benign
- Features = real-valued medical metrics like cell size, radius, etc.

Dataset source: `sklearn.datasets.load_breast_cancer()`

---

## Results

Achieved **~95% accuracy** on the test set  all with a custom-built forest!


