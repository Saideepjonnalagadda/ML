import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from LinearRegression import LinearRegression

def train_regression_model(X, y, output_description, model_save_path, regularization=0, is_multiple_output=False):
    """
    Train a linear regression model and handle both single output and multiple output scenarios.
    """
    # Split data into training and testing sets (90% train, 10% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    # Initialize the model
    model = LinearRegression()

    # Train the model
    print(f"Training {output_description} regression model...")
    model.fit(X_train, y_train, batch_size=32, max_epochs=100, patience=3, regularization=regularization)

    # Save the model weights
    model.save(model_save_path)

    # Plot the loss function (assuming loss history is tracked in the model)
    plt.plot(model.loss_history)
    plt.title(f'Training Loss over Epochs ({output_description})')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss on Validation Set')
    plt.show()

    # Evaluate the model
    test_mse = model.score(X_test, y_test)
    print(f"Test MSE for {output_description} model: {test_mse}")

    print(f"Model saved to {model_save_path}.")

def main():
    # Load the Iris dataset
    iris = load_iris()

    # First case: Single output regression
    # Predict petal width based on sepal length and width
    X_single_output = iris.data[:, :2]           # sepal length and width as features
    y_single_output = iris.data[:, 3]            # petal width as the target (single output)

    # Train model for single output
    train_regression_model(X_single_output, y_single_output,
                           output_description="Single Output (Petal Width)",
                           model_save_path='model_single_output.npz',
                           regularization=0)

    # Second case: Multiple output regression
    # Predict both petal length and petal width based on sepal length and width
    X_multiple_outputs = iris.data[:, :2]        # sepal length and width as features
    y_multiple_outputs = iris.data[:, 2:4]       # petal length and width as targets (multiple outputs)

    # Train model for multiple outputs
    train_regression_model(X_multiple_outputs, y_multiple_outputs,
                           output_description="Multiple Outputs (Petal Length and Petal Width)",
                           model_save_path='model_multiple_outputs.npz',
                           regularization=0.01)  # With regularization

if __name__ == "__main__":
    main()
