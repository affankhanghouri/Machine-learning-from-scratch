# My Custom Linear Regression 

Hey there!  
This is my own custom implementation of **Linear Regression** from scratch just using **NumPy**, no scikit-learn for the model part!

---

##  Agenda

The main reason I built this was simple:  
I wanted to understand **how scikit-learn actually works under the hood**  not just use `.fit()` and `.predict()` blindly.  
Also, I wanted to improve my **Python skills**, **logical thinking**, and really feel the math and code flow together.

---

##  What I did

-   **Wrote my own `train_test_split`** method  
-   **Wrote my own `standard scaling class`** for feature scaling    
-   Built the full **Linear Regression model manually**:
  - Did the **forward pass**
  - Computed the **gradients**
  - Applied **gradient descent**
  - Updated the **weights and bias**  
-   Used **custom exception handling** to raise clean and meaningful errors (yes, even made a `custom_exceptions.py` file )
-  Trained the model — and guess what?
  - Got **R2 score of 99.83% on test data**
  - Got **99.99%+ on training data** too!

---

##  What's Next?

In my next ML model:
- I’ll try to **improve the custom model class structure**
- And also write **sklearn-style metrics like `mean_squared_error`, `r2_score`** etc.  fully from scratch!


---

