import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from .load_data import load_data
from .fig import save_fig

def bagging_regression(X_train, X_test, Y_train, Y_test): 
   
    bag_model = BaggingRegressor(DecisionTreeRegressor(), n_estimators=500,
                            max_samples=100, n_jobs=-1, random_state=42)
    
    bag_model.fit(X_train, Y_train)
    
    y_pred_train = bag_model.predict(X_train)
    y_pred_test = bag_model.predict(X_test)
    
    # Métricas
    train_mse = mean_squared_error(Y_train, y_pred_train)
    test_mse = mean_squared_error(Y_test, y_pred_test)
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    train_mae = mean_absolute_error(Y_train, y_pred_train)
    test_mae = mean_absolute_error(Y_test, y_pred_test)
    train_r2 = r2_score(Y_train, y_pred_train)
    test_r2 = r2_score(Y_test, y_pred_test)
    
    print("BAGGING REGRESSOR completado")
    print(f"  Train MSE: {train_mse:,.2f}")
    print(f"  Test MSE:  {test_mse:,.2f}")
    print(f"  Train RMSE:  {train_rmse:,.2f}")
    print(f"  Test RMSE:  {test_rmse:,.2f}")
    print(f"  Train MAE:  {train_mae:,.2f}")
    print(f"  Test MAE:  {test_mae:,.2f}")
    print(f"  Train R²:  {train_r2:.4f}")
    print(f"  Test R²:   {test_r2:.4f}")
    
    
    return bag_model, (train_mse, test_mse, train_rmse, test_rmse, train_mae, test_mae, train_r2, test_r2)

def plot_regression_results(model, X_test, Y_test):   
   
    y_pred = model.predict(X_test)    
   
    plt.figure(figsize=(10, 6))    
    plt.scatter(Y_test, y_pred, alpha=0.6, color='royalblue', edgecolors='k', label='Datos de Prueba')
    
    min_val = min(Y_test.min(), y_pred.min())
    max_val = max(Y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Predicción')
    
    plt.xlabel("Costos Reales (Charges)")
    plt.ylabel("Costos Predichos (Charges)")
    plt.title("Regresión Bagging: Real vs Predicho")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)    
    save_fig("bagging_regression_plot")       
    plt.close()

def run_bagging_regression(X_train=None, X_test=None, Y_train=None, Y_test=None):
    
    if X_train is None:
        print("Cargando datos desde load_data")
        X_train, X_test, Y_train, Y_test = load_data()    
   
    print(f"  X_train: {X_train.shape}, Y_train: {Y_train.shape}")    
    
    model, metrics = bagging_regression(X_train, X_test, Y_train, Y_test)
    
    # Predicción ejemplo
    columnas = ['age', 'bmi', 'children', 'smoker']
    nueva_persona = pd.DataFrame([[35, 30.5, 2, 1]], columns=columnas)

    pred = model.predict(nueva_persona)[0]
    print(f"\n  Predicción ejemplo (age=35, bmi=30.5, children=2, smoker=si): ${pred:,.2f}")    
    
    plot_regression_results(model, X_test, Y_test)
    
    return model, metrics

if __name__ == "__main__":
    run_bagging_regression()
