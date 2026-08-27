Week 4 of the logistics data analysis internship focused on predictive modeling and optimization using Python. This stage builds on the work completed during Weeks 1, 2, and 3, where the logistics scenario was defined, the data was cleaned and prepared, and exploratory data analysis and visualization were performed.

The main objective of Week 4 was to move from understanding historical logistics patterns to predicting potential delivery delays. A machine learning classification approach was selected to determine whether an order is likely to be delivered late based on available order and logistics-related information.

The project uses the Brazilian E-Commerce Public Dataset by Olist. Important features considered include order value, freight value, number of items, customer state, purchase month, and purchase day of the week. The target variable is is_late, where 0 represents an on-time delivery and 1 represents a late delivery.

Two machine learning approaches were considered: Logistic Regression and Random Forest Classifier. Logistic Regression was used as an interpretable baseline, while Random Forest was selected because it can capture non-linear relationships and interactions between features.
