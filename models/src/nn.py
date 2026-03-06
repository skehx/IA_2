import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras

from .load_data import load_data
from .fig import save_fig




def build_model(architecture, input_dim):

    if architecture == 1:
        # Arquitectura 1 (Rectuangular)
        model = keras.Sequential([
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(50, activation="relu"),
            keras.layers.Dense(50, activation="relu"),
            keras.layers.Dense(1)
        ])

    elif architecture == 2:
        # Arquitectura 2 (Triangular)
        model = keras.Sequential([
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1)
        ])

    elif architecture == 3:
        # Arquitectura 3 (Piramidal)
        model = keras.Sequential([
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(1)
        ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=[keras.metrics.MeanSquaredError()]
    )

    return model


def run_nn_keras(X_train=None, X_test=None, Y_train=None, Y_test=None, epochs=30):

    if X_train is None:
        X_train, X_test, Y_train, Y_test = load_data()

    # Escalar datos
    scaler_x = StandardScaler()
    X_train_s = scaler_x.fit_transform(X_train)
    X_test_s = scaler_x.transform(X_test)

    scaler_y = StandardScaler()
    Y_train_s = scaler_y.fit_transform(Y_train.values.reshape(-1, 1))

    results = []

    for arch in [1, 2, 3]:

        print(f"\n===== Entrenando Arquitectura {arch} =====")

        model = build_model(arch, X_train_s.shape[1])

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=20,
                restore_best_weights=True
            )
        ]

        history = model.fit(
            X_train_s,
            Y_train_s,
            validation_split=0.15,
            epochs=epochs,
            batch_size=32,
            verbose=1,
            callbacks=callbacks
        )

        # Predicciones
        y_pred_train_s = model.predict(X_train_s)
        y_pred_test_s = model.predict(X_test_s)

        y_pred_train = scaler_y.inverse_transform(y_pred_train_s).ravel()
        y_pred_test = scaler_y.inverse_transform(y_pred_test_s).ravel()

        # Métricas
        train_mse = mean_squared_error(Y_train, y_pred_train)
        test_mse = mean_squared_error(Y_test, y_pred_test)
        train_r2 = r2_score(Y_train, y_pred_train)
        test_r2 = r2_score(Y_test, y_pred_test)

        print(f"Arquitectura {arch} resultado")
        print(f" Train MSE: {train_mse:,.2f} | Train R²: {train_r2:.4f}")
        print(f" Test  MSE: {test_mse:,.2f} | Test  R²: {test_r2:.4f}")

        results.append({
            "Arquitectura": arch,
            "Train MSE": train_mse,
            "Test MSE": test_mse,
            "Train R2": train_r2,
            "Test R2": test_r2
        })

        # Gráfica
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history.history["loss"], label="train")
        plt.plot(history.history["val_loss"], label="val")
        plt.title(f"Loss Arquitectura {arch}")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.scatter(Y_test, y_pred_test, alpha=0.6)
        lim = [min(Y_test.min(), y_pred_test.min()), max(Y_test.max(), y_pred_test.max())]
        plt.plot(lim, lim, "r--", lw=2)
        plt.xlabel("Valores reales")
        plt.ylabel("Predicciones")
        plt.title(f"Reales vs Predicciones (Arch {arch})")

        plt.tight_layout()
        save_fig(f"nn_arch_{arch}.png")
        plt.show()

    # Mostrar comparación final
    print("\n===== COMPARACIÓN FINAL =====")

    for r in results:
        print(
            f"Arquitectura {r['Arquitectura']} | "
            f"Train MSE: {r['Train MSE']:.2f} | "
            f"Test MSE: {r['Test MSE']:.2f} | "
            f"Train R2: {r['Train R2']:.4f} | "
            f"Test R2: {r['Test R2']:.4f}"
        )

    return results


if __name__ == "__main__":
    run_nn_keras()
