from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import joblib
import os
from embedding_generation import generate_embeddings  # Assuming your embedding generation is here
import torch
import matplotlib.pyplot as plt

# Function to explain specific indices using LIME
def apply_lime_explanation(X, model, indices_to_explain):
    """ Apply LIME explanation for selected indices. """

    # Convert embeddings into a 2D array
    explainer = LimeTabularExplainer(
        training_data=X,  # We pass the embeddings as the training data for the explainer
        mode='classification',
        class_names=model.classes_,
        discretize_continuous=True
    )

    for i in indices_to_explain:
        if i >= len(X):
            print(f"Index {i} out of bounds. Skipping.")
            continue

        # LIME expects the data to be 2D, where each sample is a row in the array
        exp = explainer.explain_instance(X[i], model.predict_proba, num_features=10)

        print(f"\nExplanation for sample {i + 1}:")
        print(exp.as_list()) # Display explanation in notebook (or use exp.as_list() for console display)

        # Generate and display the plot for the explanation
        fig = exp.as_pyplot_figure()
        plt.show()

        # Optionally, save the explanation as HTML for review
        exp.save_to_file(f'outputs/lime_explanation_{i + 1}.html')
