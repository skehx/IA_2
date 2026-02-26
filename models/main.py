from src.load_data import load_data
from src.esemble import run_bagging_regression
from src.nn import run_nn_keras

if __name__ == "__main__":    
    X_train, X_test, Y_train, Y_test = load_data()
    
    print("Ejecutando Bagging")
    run_bagging_regression(X_train, X_test, Y_train, Y_test)
    
    print("Ejecutando Neural Network")
    run_nn_keras(X_train, X_test, Y_train, Y_test)
    
   
