#  Custom Decision Tree Classifier (from Scratch)

## What this project is about

This is a beginner-friendly implementation of a **Decision Tree Classifier** written completely from scratch in Python, **without using scikit-learn’s tree module**.

The goal of this project was to **understand how decision trees work internally**, not just to get high accuracy.

So instead of using libraries like `sklearn.tree.DecisionTreeClassifier`, I built everything myself  the logic for choosing splits, calculating Gini impurity, and making predictions.

---

##  Files in this project

 

 - `My_decisonTreeClassifier.py`       ->   My custom decision tree class (core logic) 
 - `decisonTree.ipyn`       ->  Code to train and test the model on the Iris dataset 
 - `README.md`     ->   You are  reading it 🙂 


##  How the Decision Tree Works

- The tree splits the dataset based on the **best feature and threshold** that reduce **Gini impurity**.
- It keeps splitting the data until it either:
  - Hits the **max depth**
  - Finds a **pure node** (all samples belong to the same class)
- Once the tree is built, it can make predictions by **traversing down** based on input feature values.

---
