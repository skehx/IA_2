import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .load_data import load_data
from .fig import save_fig  

LOWER_BOUND = -50000
UPPER_BOUND = 50000
GENERATIONS = 50
MUTATION_RATE = 0.2
TOURNAMENT_SIZE = 3
ELITE_SIZE = 1 

# FUNCIONES PARA GRAFICAR
def plot_genetic_history(history, fold_name="final"):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(history) + 1), history, marker='o', linestyle='-', color='b', markersize=4)
    plt.title(f'Evolución del MSE por Generación ({fold_name})')
    plt.xlabel('Generaciones')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.grid(True)
    
    # Guarda la gráfica en outputs/images/ensembles/ usando tu función
    save_fig(f"evolucion_ag_{fold_name}") 
    

# FUNCIONES DEL ALGORITMO GENÉTICO 
def fitness_function(individual, X, Y):
    weights = np.array(individual[:-1]) 
    bias = individual[-1] 
    y_pred = np.dot(X, weights) + bias
    mse = np.mean((Y - y_pred) ** 2)
    return -mse

def create_population(pop_size, num_genes, lower_bound, upper_bound):
    return np.random.uniform(lower_bound, upper_bound, (pop_size, num_genes))

def tournament_selection(population, fitnesses, tournament_size=3):
    selected = []
    for _ in range(len(population)):
        contestants = np.random.choice(len(population), tournament_size, replace=False)
        best_contestant = contestants[np.argmax(fitnesses[contestants])]
        selected.append(population[best_contestant])
    return np.array(selected)

def arithmetic_crossover(parent1, parent2):
    alpha = np.random.random()
    child1 = alpha * parent1 + (1 - alpha) * parent2
    child2 = (1 - alpha) * parent1 + alpha * parent2
    return child1, child2

def mutate(individual, mutation_rate, lower_bound, upper_bound):
    if np.random.random() < mutation_rate:
        # Se cambia la media de la distribución de -2 a 0
        individual += np.random.normal(0, 2, individual.shape)
        individual = np.clip(individual, lower_bound, upper_bound)
    return individual

# FUNCIÓN DE ENTRENAMIENTO 
def run_genetic_algorithm(X_train, Y_train, pop_size, n_genes):
    """Ejecuta el AG para un fold de entrenamiento y devuelve el mejor individuo junto al historial de MSE"""
    population = create_population(pop_size, n_genes, LOWER_BOUND, UPPER_BOUND)
    
    history_mse = [] # Lista para guardar la evolución del error
    
    for generation in range(GENERATIONS):
        fitnesses = np.array([fitness_function(ind, X_train, Y_train) for ind in population])
        
        # Guardar el mejor MSE de la generación (lo pasamos a positivo)
        best_fitness_gen = np.max(fitnesses) 
        history_mse.append(-best_fitness_gen)
        
        new_population = []
        
        # Elitismo
        best_indices = np.argsort(fitnesses)[::-1][:ELITE_SIZE]
        for idx in best_indices:
            new_population.append(population[idx].copy())
            
        # Selección
        selected = tournament_selection(population, fitnesses, TOURNAMENT_SIZE)
        
        # Cruzamiento y Mutación
        while len(new_population) < pop_size:
            parent1, parent2 = random.sample(list(selected), 2)
            child1, child2 = arithmetic_crossover(parent1, parent2)
            
            child1 = mutate(child1, MUTATION_RATE, LOWER_BOUND, UPPER_BOUND)
            child2 = mutate(child2, MUTATION_RATE, LOWER_BOUND, UPPER_BOUND)
            
            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)
                
        population = np.array(new_population)
        
    # Retornar al mejor individuo de la última generación y el historial
    final_fitnesses = np.array([fitness_function(ind, X_train, Y_train) for ind in population])
    best_overall = population[np.argmax(final_fitnesses)]
    return best_overall, history_mse

# --- EJECUCIÓN CON K-FOLD, MÉTRICAS Y TESTING FINAL ---
def train_with_kfold():
    X_train, X_test, Y_train, Y_test = load_data()
    
    # Trabajaremos la validación cruzada sobre el conjunto de entrenamiento principal
    X_vals = X_train.values
    Y_vals = Y_train.values
    
    # REGLA: Población dependiente de las columnas
    num_columns = X_train.shape[1]
    POPULATION_SIZE = num_columns * 25 
    N_GENES = num_columns + 1 
    
    # Configuramos el K-Fold (por ejemplo, k=5)
    K_SPLITS = 5
    kf = KFold(n_splits=K_SPLITS, shuffle=True, random_state=42)
    
    print(f"Iniciando Validación Cruzada K-Fold ({K_SPLITS} folds)...")
    print(f"Columnas: {num_columns} | Población: {POPULATION_SIZE} | Genes: {N_GENES}\n")
    
    fold_metrics = {'mse': [], 'mae': [], 'r2': []}
    
    for fold, (train_index, val_index) in enumerate(kf.split(X_vals)):
        print(f"Entrenando Fold {fold + 1}/{K_SPLITS} ---")
        
        X_fold_train, X_fold_val = X_vals[train_index], X_vals[val_index]
        Y_fold_train, Y_fold_val = Y_vals[train_index], Y_vals[val_index]
        
        # Entrenar modelo (Ahora recibe la tupla, ignoramos el historial en cada fold con '_')
        best_individual, _ = run_genetic_algorithm(X_fold_train, Y_fold_train, POPULATION_SIZE, N_GENES)
        
        # Extraer pesos e intercepto
        best_weights = best_individual[:-1]
        best_bias = best_individual[-1]
        
        # Predecir sobre el subconjunto de validación (el fold que no vió)
        Y_fold_pred = np.dot(X_fold_val, best_weights) + best_bias
        
        # Calcular Métricas
        mse = mean_squared_error(Y_fold_val, Y_fold_pred)
        mae = mean_absolute_error(Y_fold_val, Y_fold_pred)
        r2 = r2_score(Y_fold_val, Y_fold_pred)
        
        fold_metrics['mse'].append(mse)
        fold_metrics['mae'].append(mae)
        fold_metrics['r2'].append(r2)
        
        print(f"Métricas Fold {fold + 1}: MSE = {mse:,.2f} | MAE = {mae:,.2f} | R2 = {r2:.4f}\n")
        
    # --- RESULTADOS FINALES DEL K-FOLD ---
    print("=" * 50)
    print("RESULTADOS PROMEDIO DE LA VALIDACIÓN CRUZADA (TRAIN/VAL)")
    print("=" * 50)
    print(f"MSE Promedio : {np.mean(fold_metrics['mse']):,.2f} (+/- {np.std(fold_metrics['mse']):,.2f})")
    print(f"MAE Promedio : {np.mean(fold_metrics['mae']):,.2f} (+/- {np.std(fold_metrics['mae']):,.2f})")
    print(f"R2 Promedio : {np.mean(fold_metrics['r2']):.4f} (+/- {np.std(fold_metrics['r2']):.4f})")
    print("=" * 50)
    
    print("\n" + "=" * 50)
    print("EVALUACIÓN FINAL CON CONJUNTO DE TESTING")
    print("=" * 50)
    
    # Entrenamos una última vez con todo X_train para obtener el modelo final
    print("Entrenando AG final con todo el subset de Training")
    best_final_ind, final_history = run_genetic_algorithm(X_vals, Y_vals, POPULATION_SIZE, N_GENES)
    
    # Generar y guardar la gráfica del entrenamiento final
    plot_genetic_history(final_history, fold_name="modelo_final")
    
    best_final_weights = best_final_ind[:-1]
    best_final_bias = best_final_ind[-1]
    
    # Predecimos usando los datos puros de Testing
    X_test_vals = X_test.values
    Y_test_vals = Y_test.values
    Y_test_pred = np.dot(X_test_vals, best_final_weights) + best_final_bias
    
    # Métricas de Testing
    test_mse = mean_squared_error(Y_test_vals, Y_test_pred)
    test_mae = mean_absolute_error(Y_test_vals, Y_test_pred)
    test_r2 = r2_score(Y_test_vals, Y_test_pred)
    
    print("Métricas en Testing:")
    print(f"Test MSE: {test_mse:,.2f}")
    print(f"Test MAE: {test_mae:,.2f}")
    print(f"Test R2 : {test_r2:.4f}")
    print("=" * 50)
    
    return fold_metrics, (test_mse, test_mae, test_r2)

if __name__ == '__main__': 
    train_with_kfold()
