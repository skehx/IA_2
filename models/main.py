import pandas as pd
from src.load_data import load_data
from src.esemble import run_bagging_regression
from src.nn import run_nn_keras
from src.genetic import train_with_kfold

if __name__ == "__main__": 
    # 1. Cargar Datos
    X_train, X_test, Y_train, Y_test = load_data()

    # 2. Ejecutar y capturar Bagging
    print("\n" + "="*60)
    print("1. EJECUTANDO BAGGING REGRESSOR (ENSEMBLE)")
    print("="*60)
    model_bagging, metrics_bagging = run_bagging_regression(X_train, X_test, Y_train, Y_test)
    # metrics_bagging = (train_mse, test_mse, train_rmse, test_rmse, train_mae, test_mae, train_r2, test_r2)
    bag_test_mse = metrics_bagging[1]
    bag_test_mae = metrics_bagging[5]
    bag_test_r2 = metrics_bagging[7]

    # 3. Ejecutar y capturar Red Neuronal
    print("\n" + "="*60)
    print("2. EJECUTANDO RED NEURONAL (3 ARQUITECTURAS)")
    print("="*60)
    results_nn = run_nn_keras(X_train, X_test, Y_train, Y_test)
    # Extraemos la mejor de las 3 arquitecturas (la que tenga menor Test MSE)
    best_nn = min(results_nn, key=lambda x: x['Test MSE'])

    # 4. Ejecutar y capturar Algoritmo Genético
    print("\n" + "="*60)
    print("3. EJECUTANDO ALGORITMO GENÉTICO")
    print("="*60)
    # Nota: Asegúrate de haber pegado la corrección de genetic.py que te pasé antes
    # para que devuelva estos valores correctamente
    fold_metrics, metrics_genetic = train_with_kfold()
    gen_test_mse, gen_test_mae, gen_test_r2 = metrics_genetic

    # 5. EVALUACIÓN FINAL: COMPARACIÓN DE LOS 3 MODELOS  
    print(" BENCHMARKING: VERIFICACIÓN DEL MEJOR MODELO")
   
    
    # Armamos un dataframe con los resultados de los 3 competidores
    comparativa = {
        "Modelo": [
            "Bagging Regressor", 
            f"Red Neuronal (Arch {best_nn['Arquitectura']})", 
            "Algoritmo Genético"
        ],
        "Test MSE": [bag_test_mse, best_nn['Test MSE'], gen_test_mse],
        "Test MAE": [bag_test_mae, best_nn['Test MAE'], gen_test_mae],
        "Test R²":  [bag_test_r2, best_nn['Test R2'], gen_test_r2]
    }
    
    df_resultados = pd.DataFrame(comparativa)
    
    # Formatear la tabla para que se vea bonita en consola
    df_display = df_resultados.copy()
    df_display['Test MSE'] = df_display['Test MSE'].map('{:,.2f}'.format)
    df_display['Test MAE'] = df_display['Test MAE'].map('{:,.2f}'.format)
    df_display['Test R²']  = df_display['Test R²'].map('{:.4f}'.format)
    
    print("\nTabla de Resultados Finales en Datos de Prueba (Testing):")
    print(df_display.to_string(index=False))
    
    # Lógica para elegir al ganador: El que tenga el coeficiente R² más cercano a 1
    # (o el que tenga el MSE más bajo)
    mejor_modelo_idx = df_resultados['Test R²'].idxmax()
    ganador = df_resultados.iloc[mejor_modelo_idx]
    
    print("\n" + "="*60)
    print(" CONCLUSIÓN: EL MEJOR MODELO ES:")
    print("="*60)
    print(f"El modelo ganador es: {ganador['Modelo'].upper()}")
    print(f"Logró explicar el {ganador['Test R²']*100:.2f}% de los datos (R² = {ganador['Test R²']:.4f})")
    print(f"Y tuvo el menor margen de error promedio de: ${ganador['Test MAE']:,.2f} USD")
    print("="*60)

