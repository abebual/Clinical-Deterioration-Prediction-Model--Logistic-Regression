# Clinical Deterioration Prediction Model: Logistic Regression
 
## Data
The final dataset used for the inferential statistics project includes unique ICU admission of 46,234 patients’ demographic (age), vital (blood pressure, heart rate, body temperature, and Glasgow Comma Scale), underlying conditions (HIV, metastatic cancer, and hematologic malignancy), admission type (scheduled surgical, medical, or unscheduled surgical), renal (urinary output, and Blood Urea Nitrogen), and others (serum bicarbonate level, sodium level, potassium level, and bilirubin level) data. This dataset is build based on the commonly used mortality prediction tool, Simplified Acute Physiology Score II (SAPSII).

First, run logistics regression using saps2 (the sum of all features) as explantory variable and death at ICU (hdeath - hospital death) as target variable.

## Hyperparameter Tuning
The model has some hyperparameters we can tune for hopefully better performance. In Logistic Regression, the most important parameter to tune is the regularization parameter C. Note that the regularization parameter is not always part of the logistic regression model. The regularization parameter is used to control for unlikely high regression coefficients, and in other cases can be used when data is sparse, as a method of feature selection. We may not need this for our model but worth checking.
Using the cv_score function (5-fold cross validation) for a basic logistic regression model without regularization,the score on the held-out data (test data) is 0.908, 91%.

Based on the training set the best model parameter is 0.9079345258458951 for a C value of 0.01.
Running the model with C=0.01 gives as the same accuracy results on the test data as the deafult. This is not always the case hence important to experment with the hyperparameters that works best with new data.

Best score on training data: 0.9080219037022492 using {'C': 0.1, 'penalty': 'l2', 'solver': 'liblinear'}. It gives a diffrent best value of C - this time 0.1. The GridSearchCV performs slightly better on test data (0.9036 vs 0.9044), almost the same.

Let's first set some code up for classification that we will need for further discussion on the math. We first set up a function cv_optimize which takes a classifier clf, a grid of hyperparameters (such as a complexity parameter or regularization parameter) implemented as a dictionary parameters, a training set (as a samples x features array) Xtrain, and a set of labels ytrain. The code takes the traning set, splits it into n_folds parts, sets up n_folds folds, and carries out a cross-validation by splitting the training set into a training and validation section for each foldfor us. It prints the best value of the parameters, and retuens the best classifier to us.

## Cross Validation Score
we should evaluate the performance of an algorithm rigorously by using resampling approaches (e.g. 100 times 5-fold cross-validation) to get some measurement of the variability in the performance of the algorithm. Maybe on a particular hold-out set, two algorithms have very similar performance but the variability of their estimates is massively different. That has serious implication on when we deploy our model in the future or use it to draw conclusion about future performance. 
## Plotting an ROC curve - receiver operating characteristic
## Weighted Logistic Regression for Imbalanced Dataset

## XGBoost
XGBoost Python api provides a method to assess the incremental performance by the incremental number of trees. It uses two arguments: “eval_set” — usually Train and Test sets — and the associated “eval_metric” to measure your error on these evaluation sets.

