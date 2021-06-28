# Supervised Learning on Online Shoppers Purchasing Intention Data

### Project for Statistical Learning Course of MSc in Data Science for Management (UniCT).

The work in this project is aimed at building a predictive model in the context of a classification problem: specifically, it is about predicting the purchase intention of e-commerce web pages’ visitors, based on their behavior during the browsing session.

For this purpose, once the predictor variables have been identified, three main supervised learning methods are considered:
1. Logistic Regression;
2. Support Vector Machines;
3. Neural Networks.

Each model is trained and evaluated. Then the performance of the defined models is compared on the basis of their misclassification rate and finally the best of them is tested to make predictions of the response variable *Revenue*.

The dataset used consists of feature vectors belonging to 12330 sessions, and was formed so that each session would belong to a different user in a 1-year period to avoid any tendency to a specific campaign, special day, user profile, or period. The dataset contains 10 numerical and 8 categorical attributes, of which the *Revenue* attribute is used as the class label.

The original dataset is available here: https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset#

For the analysis purposes, it was split in:

* Train set: about 60% of the units of the original dataset.
* Validation set: about 20% of the units of the original dataset.
* Test set: about 20% of the units of the original dataset.
