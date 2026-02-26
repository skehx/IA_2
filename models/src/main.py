from .load_data import load_data
from .esemble import run_bagging_regression
from .nn import run_nn_keras
if __name__ == "__main__":    
    X_train, X_test, Y_train, Y_test = load_data()
    
    print("Ejecutando Bagging")
    run_bagging_regression(X_train, X_test, Y_train, Y_test)
    
    print("Ejecutando Neural net")
    run_nn_keras(X_train, X_test, Y_train, Y_test)
   
