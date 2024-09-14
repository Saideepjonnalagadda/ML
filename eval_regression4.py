import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from LinearRegression import LinearRegression

def evaluate_regression_model(X, y, output_type, model_path):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    model = LinearRegression()
    model.load(model_path)
    # Evaluation
    test_mse = model.score(X_test, y_test)
    print(f"Test MSE for {output_type} : {test_mse}")

def main():
    iris = load_iris()
    #Single output evaluation
    X_single = iris.data[:, 2:4]           
    y_single = iris.data[:, 1]            
    evaluate_regression_model(X_single, y_single,
                              output_type="Single Output-Sepal length)",
                              model_path='model_single_output.npz')
if __name__ == "__main__":
    main()
