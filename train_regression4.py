import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from LinearRegression import LinearRegression

def train_regression(X, y, output_type, save_path, regularization=0, is_multiple_output=False):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    model = LinearRegression()
    print(f" {output_type} : Training regression model")
    model.fit(X_train, y_train, regularization=regularization)
    model.save(save_path)
    plt.plot(model.loss)
    plt.title(f'Training Loss over Epochs ({output_type})')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss on training Set')
    plt.show()

    # Evaluation
    test_mse = model.score(X_test, y_test)
    print(f"Test MSE for model {output_type} : {test_mse}")

    print(f"Model saved at file location {save_path}.")

def main():
    iris = load_iris()    
    # Single output regression - Predicting sepal length based on petal length and width
    X_single = iris.data[:, 2:4]           
    y_single = iris.data[:, 1]            
    train_regression(X_single, y_single,
                           output_type="Single Output - sepal width",
                           save_path='model_single_output.npz',
                           regularization=0)

    # Single output regression with L2 regularization- Predicting sepal length based on petal length and width
    X_single = iris.data[:, 2:4]           
    y_single = iris.data[:, 1]            
    train_regression(X_single, y_single,
                           output_type="L2 regularized Single Output - sepal width",
                           save_path='model_single_output.npz',
                           regularization=0.4)  

if __name__ == "__main__":
    main()