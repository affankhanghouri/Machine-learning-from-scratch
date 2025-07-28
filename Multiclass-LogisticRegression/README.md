#  Custom Multi-Class Logistic Regression (From Scratch)

This project is all about building a **Multi-Class Logistic Regression classifier** from the ground up using only `NumPy` and pure Python. No scikit-learn models involved!

---

##  Goal of the Project

The main purpose of this project was **learning how Logistic Regression actually works** internally, especially for **multi-class classification** problems.

Instead of using prebuilt models, I wrote the logic myself including:
- Softmax function
- Cross-entropy loss
- Gradient Descent optimization
- One-vs-All classification

This helped me understand every single line of how multi-class classification is really done.

---

##  Dataset Used: Iris

- The famous **Iris dataset** from scikit-learn.
- 3 classes: Setosa, Versicolor, Virginica.
- 4 numerical features (sepal & petal length/width).
- Small and balanced — so perfect for learning and fast testing.

---

##  Why Accuracy is 100%

Since the **Iris dataset is small and clean**, and because of the simplicity of logistic regression, the model was able to achieve **100% accuracy** on this particular dataset.

But in real-world scenarios, results won’t be this perfect  overfitting or class imbalance can cause much lower performance.

---



Here’s what the code does:

1. **Loads the dataset**
   - Using `sklearn.datasets.load_iris`
2. **One-hot encodes** the labels manually
3. **Implements logistic regression**
   - With multi-class support using softmax
4. **Defines forward pass, loss, and backward pass**
   - Cross-entropy loss
   - Gradient descent update rule
5. **Trains the model** over multiple epochs
6. **Evaluates the accuracy**

--
