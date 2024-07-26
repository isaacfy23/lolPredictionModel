from pickle import load
from pandas import read_csv
import os
import numpy as np
import graphviz
from sklearn.tree import export_graphviz
import pydotplus
from IPython.display import Image


def load_model(filename):
    absPath = os.path.join("../models", filename)
    with open(absPath, "rb") as file:
        model = load(file)
    return model


modelsNames = {
    "1": "best_model.pkl",
}


test_data = read_csv("../data/processed/processed_match_data_test.csv")

random_index = np.random.choice(test_data.index)
random_row = test_data.loc[[random_index]]
print(random_row)

for modelName in modelsNames.values():
    model = load_model(modelName)
    prediction = model.predict(random_row.drop("matchID", axis=1))
    print(prediction, modelName)

"""
if modelName == "Random Forest.pkl":
    estimator = model.estimators_[0]

    # Export the tree to a dot file with limited depth and top-down orientation
    dot_data = export_graphviz(
        estimator,
        out_file=None,
        feature_names=test_data.drop("matchID", axis=1).columns,
        class_names=["Loss", "Win"],
        filled=True,
        rounded=True,
        special_characters=True,
        max_depth=3,
        label='none',
        impurity=False,
        proportion=False,
        precision=2
    )

    # Use pydotplus to create an image from the dot file
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png("plots/tree_but_like_chill.png")

    # Display the tree image
    Image(graph.create_png())

"""