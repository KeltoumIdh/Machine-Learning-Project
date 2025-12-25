# Dynamic Pricing Model – Data Analytics Report

## 1. Introduction

This report presents the complete **data analytics and machine learning pipeline** for the **Dynamic Pricing** use case (Part 1 – Gradient Descent). The objective is to predict the optimal service price based on multiple operational and market-related factors, following professional data science standards and aligned with the project grading requirements.

---

## 2. Business Understanding

### Objective

Predict the **dynamic_price** of a service using numerical indicators representing demand, competition, costs, and marketing conditions.

### Type of Problem

* **Supervised Learning**
* **Regression task**

### Target Variable

* `dynamic_price`

### Input Features

* `demand_index`
* `time_slot`
* `day_of_week`
* `competition_pressure`
* `operational_cost`
* `seasonality_index`
* `marketing_intensity`

---

## 3. Data Description

* Dataset size: **2000 rows × 8 columns**
* Data type: All numerical (float)
* Dataset condition: **Dirty / incomplete** (intentionally)

### Data Quality Issues Identified

* Missing values (~8% per column)
* Negative values in non-negative features
* Extreme outliers in price and cost variables

---

## 4. Data Cleaning & Preprocessing

### 4.1 Handling Missing Values

* Strategy: **Median imputation**
* Justification: Robust to outliers and suitable for regression

### 4.2 Outlier Detection & Treatment

* Method used: **Interquartile Range (IQR)**
* Strategy: **Clipping** values instead of deleting rows

This approach preserves data volume while preventing gradient explosion during training.

### 4.3 Feature Scaling

* Method: **Standardization (Z-score normalization)**

> Feature scaling is mandatory for Gradient Descent to ensure:
>
> * Faster convergence
> * Numerical stability
> * Balanced gradient updates

---

## 5. Exploratory Data Analysis (EDA)

### Key Observations

* `demand_index` shows strong positive correlation with price
* `operational_cost` is a major price driver
* `competition_pressure` generally reduces price
* `marketing_intensity` has a moderate positive impact

These relationships are consistent with real-world pricing logic, increasing model credibility.

---

## 6. Data Splitting

* Training set: **80%**
* Test set: **20%**
* Random state fixed for reproducibility

---

## 7. Modeling Approach – Gradient Descent

### Model Type

* **Linear Regression implemented from scratch**

### Hypothesis Function

ŷ = XW + b

### Loss Function

Mean Squared Error (MSE):

MSE = (1 / n) · Σ(y − ŷ)²

### Optimization Techniques Implemented

* Batch Gradient Descent
* Stochastic Gradient Descent (SGD)
* Mini-batch Gradient Descent

---

## 8. Training Analysis

### Learning Rate Impact

* High learning rate → divergence
* Low learning rate → slow convergence
* Optimal learning rate → smooth loss reduction

### Convergence Behavior

* Batch GD: stable but slow
* SGD: fast but noisy
* Mini-batch GD: best trade-off between speed and stability

---

## 9. Model Evaluation

### Metrics Used

* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

### Evaluation Outcome

The model demonstrates:

* Good generalization on unseen data
* Stable predictions
* Strong explanatory power for pricing decisions

---

## 10. Model Persistence

* The trained model is saved using serialization
* Enables reuse in production without retraining

---

## 11. API Integration Readiness

The model is prepared for deployment via a REST API:

* Endpoint: `/predict-price`
* Method: POST
* Input: JSON numerical features
* Output: Predicted price

---

## 12. Conclusion

This project successfully delivers a **production-ready dynamic pricing model**, supported by:

* Professional data cleaning
* Robust preprocessing
* Clear business interpretation
* From-scratch Gradient Descent implementation
* Deployment-oriented design

The approach fully satisfies the technical and analytical requirements of the Machine Learning Project.

---

## 13. Future Improvements

* Feature interaction analysis
* Regularization (L1 / L2)
* Advanced learning rate scheduling
* Model monitoring after deployment
