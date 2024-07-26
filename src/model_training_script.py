import pandas as pd
import numpy as np

# machine learning
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from pickle import dump

train_data_diff = pd.read_csv('../data/processed/processed_match_data_train.csv')

X_train = train_data_diff.drop("blueWin", axis=1)
Y_train = train_data_diff["blueWin"]

# Define search space for hyperparameters

# param_grid_knn = {
#     'n_neighbors': [1, 3, 5, 10],
#     'weights': ['uniform', 'distance'],
#     'metric': ['euclidean', 'manhattan']
# }

param_grid_rf = {
    'n_estimators': [250, 275],
    'max_depth': [20, 30, 40],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [4, 5, 6]
}

# grid_search_knn = GridSearchCV(estimator=KNeighborsClassifier(),
#                                param_grid=param_grid_knn,
#                                cv=5,
#                                n_jobs=-1,
#                                scoring='accuracy')


print("Random Forest Grid Search starting...")
grid_search_rf = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                              param_grid=param_grid_rf,
                              cv=5,
                              n_jobs=-1,
                              scoring='accuracy')

# grid_search_knn.fit(X_train, Y_train)
grid_search_rf.fit(X_train, Y_train)

# # KNN
# best_knn = grid_search_knn.best_estimator_
# knn_accuracy = round(best_knn.score(X_train, Y_train) * 100, 2)
# print(f'Best KNN parameters: {grid_search_knn.best_params_}')
# print(f'KNN accuracy: {knn_accuracy}')

# Random Forest
best_rf = grid_search_rf.best_estimator_
rf_accuracy = round(best_rf.score(X_train, Y_train) * 100, 2)
print(f'Best Random Forest parameters: {grid_search_rf.best_params_}')
print(f'Random Forest accuracy: {rf_accuracy}')

with open('../models/Random Forest.pkl', 'wb') as f:
    dump(best_rf, f)

# models_dict = {
#     "KNN": KNeighborsClassifier(n_neighbors=2,
#                                 weights='uniform'),
#     "Random Forest": RandomForestClassifier(n_estimators=100,
#                                             max_depth=50,
#                                             min_samples_split=6,
#                                             min_samples_leaf=2,
#                                             random_state=42)
# }
# # optimize parameters and explain why
#
# models = pd.DataFrame({
#     "Model": [],
#     "Score": []
#     })
#
# # Train the models and store their scores in the DataFrame
# for model_name, model in models_dict.items():
#     model.fit(X_train, Y_train)
#     score = round(model.score(X_train, Y_train) * 100, 2)
#     models = models._append({"Model": model_name, "Score": score}, ignore_index=True)
#     print(model_name, "successfully trained")
#
#     # with open(f'./models/{model_name}.pkl', 'wb') as f:
#     #     dump(model, f)
#
# print(models)