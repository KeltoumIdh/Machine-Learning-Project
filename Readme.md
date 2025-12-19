PART 1 – Pricing
- Remove negative demand
- Fill missing numerical values with median
- Standardize features

PART 2 – Tickets
- Drop rows with missing text
- Lowercase + remove punctuation
- TF-IDF vectorization

PART 3 – Credit Risk
- Drop missing target
- Fill numerical features with median
- Limit tree depth to avoid overfitting

models:
##Pricing:
“I implemented linear regression using gradient descent from scratch.
At each iteration, I compute the prediction error, calculate the gradients,
and update the weights to minimize the mean squared error.”

“I implemented linear regression using three Gradient Descent variants from scratch.
I evaluated them using MSE and R² and selected Mini-Batch Gradient Descent for its stable convergence.
The trained model is then saved and reused in an API.”

##tickets
“I preprocessed ticket text, converted it to TF-IDF vectors,
then trained two multiclass logistic regression approaches: One-vs-Rest and Softmax.
I compared their performance and saved the best model.”

##credit
“I used RandomizedSearch with cross-validation to tune the Decision Tree hyperparameters.
I compared Gini and Entropy criteria, controlled overfitting through depth and leaf constraints,
and selected the best model based on test accuracy.”