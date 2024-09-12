import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from LinearRegression import LinearRegression

def evaluate_regression_model(X, y, output_description, model_load_path):
    """
    Load a previously trained model and evaluate it on the given data.
    """
    # Split data into training and testing sets
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.1, random_state=42)  # Only need the test set

    # Load the model
    model = LinearRegression()
    model.load(model_load_path)

    # Evaluate performance
    test_mse = model.score(X_test, y_test)
    print(f"Test MSE for {output_description}: {test_mse}")

def main():
    # Load the Iris dataset
    iris = load_iris()

    # First case: Single output regression evaluation
    X_single_output = iris.data[:, :2]           # sepal length and width as features
    y_single_output = iris.data[:, 3]            # petal width as the output (single output)
    
    evaluate_regression_model(X_single_output, y_single_output,
                              output_description="Single Output (Petal Width)",
                              model_load_path='model_single_output.npz')

    # Second case: Multiple output regression evaluation
    X_multiple_outputs = iris.data[:, :2]        # sepal length and width as features
    y_multiple_outputs = iris.data[:, 2:4]       # petal length and petal width as outputs (multiple outputs)
    
    evaluate_regression_model(X_multiple_outputs, y_multiple_outputs,
                              output_description="Multiple Outputs (Petal Length and Petal Width)",
                              model_load_path='model_multiple_outputs.npz')

if __name__ == "__main__":
    main()
