import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import importlib
import time
import logging
from statistics import stdev
import psutil  # For memory and CPU usage
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
from tqdm import tqdm

# Configure logging to write to stdout
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

# Define test objective function (noise-free, quadratic)
def objective_function(x, dim=10):
    """Simple quadratic function: sum of squares."""
    if not isinstance(x, np.ndarray):
        x = np.array(x, dtype=np.float64)
    return np.sum(x**2)

# Define noisy objective function for robustness test
def noisy_objective_function(x, dim=10, noise_level=0.1):
    """Quadratic function with added Gaussian noise."""
    true_value = objective_function(x, dim)
    noise = np.random.normal(0, noise_level * (true_value + 1e-10))
    return true_value + noise

# SVM Hyperparameter Tuning Objective Function
def extract_svm_hyperparameter_schema():
    """Extract hyperparameter schema for SVR."""
    schema = {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'C': (0.1, 10.0),
        'epsilon': (0.01, 1.0),
    }
    return schema

def decode_svm_solution(solution, schema):
    """Decode a solution into SVR hyperparameters."""
    if isinstance(solution, np.ndarray):
        solution = solution.tolist()
    elif isinstance(solution, tuple):
        solution = list(solution)
    param_keys = list(schema.keys())
    if len(solution) != len(param_keys):
        logging.warning(f"Solution size {len(solution)} doesn't match hyperparameter count {len(param_keys)}")
        return {}
    decoded_solution = {}
    for i, param in enumerate(param_keys):
        if i >= len(solution):
            continue
        value = solution[i]
        if np.isnan(value):
            logging.warning(f"{param} is NaN, using default value")
            decoded_solution[param] = schema[param][0] if isinstance(schema[param], list) else schema[param][0]
            continue
        options = schema[param]
        if isinstance(options, tuple):
            min_val, max_val = options
            scaled_value = min_val + (max_val - min_val) * (value / (len(param_keys) - 1))
            decoded_solution[param] = max(min_val, min(max_val, scaled_value))
        else:
            index = int(round(min(max(value, 0), len(options) - 1)))
            decoded_solution[param] = options[index]
    return decoded_solution

# Counter for solution evaluations to reduce log verbosity
solution_count = [0]

def svm_objective_function(solution, dim=3):
    """Objective function for SVM hyperparameter tuning."""
    solution_count[0] += 1
    # Log every 100th solution to reduce clutter
    if solution_count[0] % 100 == 0:
        logging.info(f"Received solution #{solution_count[0]}: {solution}, type: {type(solution)}, shape: {getattr(solution, 'shape', 'N/A')}")
    if not isinstance(solution, (list, np.ndarray)):
        logging.error(f"Solution is not a valid list or array! Type: {type(solution)}")
        return float('inf')
    if isinstance(solution, np.ndarray):
        solution = solution.tolist()
    if len(solution) != dim:
        logging.error(f"Solution length {len(solution)} does not match expected dim {dim}")
        return float('inf')
    schema = extract_svm_hyperparameter_schema()
    decoded_params = decode_svm_solution(solution, schema)
    if not decoded_params:
        return float('inf')
    X, y = make_regression(n_samples=200, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    try:
        model = SVR()
        model.set_params(**decoded_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return mean_squared_error(y_test, y_pred)
    except Exception as e:
        logging.error(f"Error during SVM training: {e}")
        return float('inf')

# Evaluation aspects
ASPECTS = [
    'Accuracy', 'Time Efficiency', 'Convergence Speed', 'Scalability',
    'Memory Usage', 'Robustness to Noise', 'Sensitivity to Parameters',
    'Adaptability to Problem Complexity', 'Energy Efficiency',
    'Solution Stability Over Iterations', 'Exploration vs. Exploitation Balance',
    'Parallelization Efficiency', 'Ease of Use'
]

PLOT_ASPECTS = [
    'Accuracy', 'Time Efficiency', 'Convergence Speed', 'Scalability',
    'Memory Usage', 'Robustness to Noise', 'Sensitivity to Parameters',
    'Adaptability to Problem Complexity', 'Energy Efficiency',
    'Solution Stability Over Iterations', 'Exploration vs. Exploitation Balance',
    'Parallelization Efficiency'
]

def evaluate_wrapper(wrapper_module, module_name, dim=10, runs=3, method="DISO", overall_progress=None):
    """Evaluate a Wrapper on the test objective function."""
    np.random.seed(hash(module_name) % 2**32)
    bounds = [-5.0, 5.0] * dim
    population_size = 20
    max_iter = 50

    fitness_scores, run_times, memory_usages = [], [], []
    convergence_times, noisy_fitness_scores, param_sensitivity_fitness = [], [], []
    svm_fitness_scores, cpu_usages, short_run_fitnesses = [], [], []
    initial_fitness, parallel_fitness, parallel_run_times = None, [], []

    # Reset solution count for this module
    solution_count[0] = 0

    for i in tqdm(range(runs), desc=f"Evaluating {module_name}", file=sys.stderr, leave=False):
        try:
            if initial_fitness is None:
                wrapper = wrapper_module.Wrapper(dim, population_size, max_iter, bounds, method=method)
                wrapper.optimize(lambda x: objective_function(x, dim=dim))
                _, initial_fitness = wrapper.get_best_solution()
                initial_fitness += np.random.uniform(-0.5, 0.5)
                wrapper.free()
                logging.info(f"Run {i+1}/{runs}: Initial fitness baseline set for {module_name}")
                if overall_progress:
                    overall_progress.update(1)

            wrapper = wrapper_module.Wrapper(dim, population_size, max_iter, bounds, method=method)
            start_time = time.time()
            process = psutil.Process()
            mem_before = process.memory_info().rss
            wrapper.optimize(lambda x: objective_function(x, dim=dim))
            _, fitness = wrapper.get_best_solution()
            fitness += np.random.uniform(-0.5, 0.5)
            elapsed_time = time.time() - start_time
            cpu_usage = process.cpu_percent(interval=None)
            mem_after = process.memory_info().rss
            logging.info(f"Run {i+1}/{runs}: Standard run completed for {module_name}, fitness={fitness}")
            fitness_scores.append(max(0, fitness))
            run_times.append(elapsed_time)
            memory_usages.append(mem_after - mem_before)
            cpu_usages.append(cpu_usage * elapsed_time)
            convergence_times.append(1.0 if fitness < 0.5 else max(0, 1 - elapsed_time / 3.0))
            wrapper.free()
            if overall_progress:
                overall_progress.update(1)

            wrapper = wrapper_module.Wrapper(dim, population_size, max_iter, bounds, method=method)
            wrapper.optimize(lambda x: noisy_objective_function(x, dim=dim, noise_level=0.1))
            _, noisy_fitness = wrapper.get_best_solution()
            noisy_fitness += np.random.uniform(-0.5, 0.5)
            logging.info(f"Run {i+1}/{runs}: Noisy run completed for {module_name}, fitness={noisy_fitness}")
            noisy_fitness_scores.append(max(0, noisy_fitness))
            wrapper.free()
            if overall_progress:
                overall_progress.update(1)

            wrapper = wrapper_module.Wrapper(dim, population_size // 2, max_iter, bounds, method=method)
            wrapper.optimize(lambda x: objective_function(x, dim=dim))
            _, param_fitness = wrapper.get_best_solution()
            param_fitness += np.random.uniform(-0.5, 0.5)
            logging.info(f"Run {i+1}/{runs}: Sensitivity run completed for {module_name}, fitness={param_fitness}")
            param_sensitivity_fitness.append(max(0, param_fitness))
            wrapper.free()
            if overall_progress:
                overall_progress.update(1)

            svm_dim = 3
            svm_bounds = [(0, 3), (0, 3), (0, 3)]
            wrapper = wrapper_module.Wrapper(svm_dim, population_size, max_iter, svm_bounds, method=method)
            wrapper.optimize(lambda x: svm_objective_function(x, dim=svm_dim))
            _, svm_fitness = wrapper.get_best_solution()
            svm_fitness += np.random.uniform(-0.5, 0.5)
            logging.info(f"Run {i+1}/{runs}: SVM tuning run completed for {module_name}, fitness={svm_fitness}")
            svm_fitness_scores.append(max(0, svm_fitness))
            wrapper.free()
            if overall_progress:
                overall_progress.update(1)

            wrapper = wrapper_module.Wrapper(dim, population_size, max_iter // 5, bounds, method=method)
            wrapper.optimize(lambda x: objective_function(x, dim=dim))
            _, short_fitness = wrapper.get_best_solution()
            short_fitness += np.random.uniform(-0.5, 0.5)
            logging.info(f"Run {i+1}/{runs}: Short run completed for {module_name}, fitness={short_fitness}")
            short_run_fitnesses.append(max(0, short_fitness))
            wrapper.free()
            if overall_progress:
                overall_progress.update(1)

            wrapper = wrapper_module.Wrapper(dim, population_size * 2, max_iter, bounds, method=method)
            start_time = time.time()
            wrapper.optimize(lambda x: objective_function(x, dim=dim))
            _, parallel_fit = wrapper.get_best_solution()
            parallel_fit += np.random.uniform(-0.5, 0.5)
            parallel_run_time = time.time() - start_time
            logging.info(f"Run {i+1}/{runs}: Parallel run completed for {module_name}, fitness={parallel_fit}")
            parallel_fitness.append(max(0, parallel_fit))
            parallel_run_times.append(parallel_run_time)
            wrapper.free()
            if overall_progress:
                overall_progress.update(1)

        except Exception as e:
            logging.error(f"Run {i+1}/{runs}: Error evaluating {module_name}: {e}")
            return None

    min_fitness = min(fitness_scores)
    max_fitness = max(fitness_scores) if max(fitness_scores) > min_fitness else min_fitness + 1
    accuracy = 1.0 - (np.mean(fitness_scores) - min_fitness) / (max_fitness - min_fitness + 1e-10)

    min_time = min(run_times)
    max_time = max(run_times) if max(run_times) > min_time else min_time + 1
    time_efficiency = 1.0 - (np.mean(run_times) - min_time) / (max_time - min_time + 1e-10)

    min_mem = min(memory_usages)
    max_mem = max(memory_usages) if max(memory_usages) > min_mem else min_mem + 1
    memory_usage = 1.0 - (np.mean(memory_usages) - min_mem) / (max_mem - min_mem + 1e-10)

    convergence_speed = np.mean(convergence_times)
    min_noisy_fitness = min(noisy_fitness_scores)
    max_noisy_fitness = max(noisy_fitness_scores) if max(noisy_fitness_scores) > min_noisy_fitness else min_noisy_fitness + 1
    robustness = 1.0 - (np.mean(noisy_fitness_scores) - min_noisy_fitness) / (max_noisy_fitness - min_noisy_fitness + 1e-10)

    min_param_fitness = min(param_sensitivity_fitness)
    max_param_fitness = max(param_sensitivity_fitness) if max(param_sensitivity_fitness) > min_param_fitness else min_param_fitness + 1
    sensitivity = 1.0 - (np.mean(param_sensitivity_fitness) - min_param_fitness) / (max_param_fitness - min_param_fitness + 1e-10)

    min_svm_fitness = min(svm_fitness_scores)
    max_svm_fitness = max(svm_fitness_scores) if max(svm_fitness_scores) > min_svm_fitness else min_svm_fitness + 1
    adaptability = 1.0 - (np.mean(svm_fitness_scores) - min_svm_fitness) / (max_svm_fitness - min_svm_fitness + 1e-10)

    min_energy = min(cpu_usages)
    max_energy = max(cpu_usages) if max(cpu_usages) > min_energy else min_energy + 1
    energy_efficiency = 1.0 - (np.mean(cpu_usages) - min_energy) / (max_energy - min_energy + 1e-10)

    short_fitness_std = stdev(short_run_fitnesses) if len(short_run_fitnesses) > 1 else 0
    short_fitness_mean = np.mean(short_run_fitnesses) + 1e-10
    solution_stability = max(0, 1 - short_fitness_std / short_fitness_mean)

    final_fitness = np.mean(fitness_scores)
    exploration_balance = max(0, 1 - (final_fitness - min_fitness) / (initial_fitness - min_fitness + 1e-10)) if initial_fitness > min_fitness else 0.5

    min_parallel_fit = min(parallel_fitness)
    max_parallel_fit = max(parallel_fitness) if max(parallel_fitness) > min_parallel_fit else min_parallel_fit + 1
    parallel_efficiency = 1.0 - (np.mean(parallel_fitness) - min_parallel_fit) / (max_parallel_fit - min_parallel_fit + 1e-10)
    min_parallel_time = min(parallel_run_times)
    max_parallel_time = max(parallel_run_times) if max(parallel_run_times) > min_parallel_time else min_parallel_time + 1
    parallel_efficiency = max(parallel_efficiency, 1.0 - (np.mean(parallel_run_times) - min_parallel_time) / (max_parallel_time - min_parallel_time + 1e-10))

    try:
        wrapper = wrapper_module.Wrapper(dim * 2, population_size, max_iter, bounds * 2, method=method)
        start_time = time.time()
        wrapper.optimize(lambda x: objective_function(x, dim=dim*2))
        scalability_time = time.time() - start_time
        min_scalability = min(scalability_time, np.mean(run_times) * 4)
        max_scalability = max(scalability_time, np.mean(run_times) * 4)
        scalability = 1.0 - (scalability_time - min_scalability) / (max_scalability - min_scalability + 1e-10)
        wrapper.free()
        if overall_progress:
            overall_progress.update(1)
    except Exception as e:
        logging.error(f"Scalability test failed for {module_name}: {e}")
        scalability = 0.5
        if overall_progress:
            overall_progress.update(1)

    ease_of_use = 1.0
    return {
        'Accuracy': accuracy, 'Time Efficiency': time_efficiency,
        'Convergence Speed': convergence_speed, 'Scalability': scalability,
        'Memory Usage': memory_usage, 'Robustness to Noise': robustness,
        'Sensitivity to Parameters': sensitivity,
        'Adaptability to Problem Complexity': adaptability,
        'Energy Efficiency': energy_efficiency,
        'Solution Stability Over Iterations': solution_stability,
        'Exploration vs. Exploitation Balance': exploration_balance,
        'Parallelization Efficiency': parallel_efficiency,
        'Ease of Use': ease_of_use
    }

def plot_individual_spider_guage(scores, method_name, save_path):
    angles = np.linspace(0, 2 * np.pi, len(PLOT_ASPECTS), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
    values = [scores[aspect] for aspect in PLOT_ASPECTS] + [scores[PLOT_ASPECTS[0]]]
    ax.fill(angles, values, color='blue', alpha=0.25)
    ax.plot(angles, values, color='blue', linewidth=2, label=method_name)
    for i, (angle, value) in enumerate(zip(angles[:-1], values[:-1])):
        ax.text(angle, value, f'{value:.2f}', ha='center', va='bottom', fontsize=8)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(PLOT_ASPECTS, fontsize=8, wrap=True)
    ax.set_title(f'Performance of {method_name}', size=15, color='blue', y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_overall_spider_guage(all_scores, save_path):
    angles = np.linspace(0, 2 * np.pi, len(PLOT_ASPECTS), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(polar=True))
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    max_score = max(scores[aspect] for scores in all_scores.values() for aspect in PLOT_ASPECTS)
    for (method_name, scores), color in zip(all_scores.items(), colors):
        values = [scores[aspect] for aspect in PLOT_ASPECTS] + [scores[PLOT_ASPECTS[0]]]
        ax.fill(angles, values, color=color, alpha=0.1)
        ax.plot(angles, values, color=color, linewidth=2, label=method_name)
        for i, (angle, value) in enumerate(zip(angles[:-1], values[:-1])):
            ax.text(angle, value, f'{value:.2f}', ha='center', va='bottom', fontsize=8)
    ax.set_ylim(0, max_score * 1.1)
    ax.set_yticks(np.linspace(0, max_score * 1.1, 6))
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(PLOT_ASPECTS, fontsize=8, wrap=True)
    ax.set_title('Overall Performance Comparison', size=15, y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def main(method="DISO"):
    # Test tqdm to confirm it works in this environment
    print("Testing progress bar functionality...", file=sys.stderr)
    for _ in tqdm(range(5), desc="Test Progress Bar", file=sys.stderr):
        time.sleep(0.1)  # Simulate some work
    print("Test complete. Starting benchmark...", file=sys.stderr)

    module_dirs = ['CIntegration_cffi', 'CIntegration_cffi1', 'CIntegration_cffi2', 'CIntegration_cffi3']
    for module_dir in module_dirs:
        init_path = Path(module_dir) / '__init__.py'
        if not init_path.exists():
            init_path.touch()

    # Calculate total tasks
    runs_per_module = 3  # Number of runs per module
    tasks_per_run = 7    # Number of optimize calls per run (initial, standard, noisy, sensitivity, svm, short, parallel)
    scalability_tasks = 1  # Scalability test per module
    num_modules = len(module_dirs)
    total_tasks = num_modules * (runs_per_module * tasks_per_run + scalability_tasks)

    original_cwd = os.getcwd()
    all_scores = {}
    modules = ['CIntegration_cffi.wrapper', 'CIntegration_cffi1.wrapper', 'CIntegration_cffi2.wrapper', 'CIntegration_cffi3.wrapper']
    
    # Overall progress bar
    with tqdm(total=total_tasks, desc="Overall Benchmark Progress", file=sys.stderr) as overall_progress:
        for module_path in tqdm(modules, desc="Evaluating Modules", file=sys.stderr):
            module_name = module_path.split('.')[0]
            module_dir = os.path.join(original_cwd, module_name)
            try:
                os.chdir(module_dir)
                logging.info(f"Changed working directory to {module_dir}")
                wrapper_module = importlib.import_module(module_path)
                scores = evaluate_wrapper(wrapper_module, module_name, method=method, overall_progress=overall_progress)
                if scores:
                    all_scores[module_name] = scores
                else:
                    logging.warning(f"Skipping {module_name} due to evaluation failure")
            except ImportError as e:
                logging.error(f"Failed to import {module_path}: {e}")
            except Exception as e:
                logging.error(f"Error processing {module_name}: {e}")
            finally:
                os.chdir(original_cwd)
                logging.info(f"Reverted working directory to {original_cwd}")

    for method_name, scores in all_scores.items():
        save_path = f"{method_name}_spider.png"
        plot_individual_spider_guage(scores, method_name, save_path)
        logging.info(f"Saved individual spider graph for {method_name} at {save_path}")

    if all_scores:
        save_path = "overall_performance_spider.png"
        plot_overall_spider_guage(all_scores, save_path)
        logging.info(f"Saved overall performance spider graph at {save_path}")

if __name__ == "__main__":
    main(method="DISO")
