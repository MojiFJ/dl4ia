import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from load_auto import load_auto
from utils import *

# Loading and preprocessing the data
features = ['horsepower', 'cylinders', 'displacement', 'weight', 'acceleration', 'year', 'origin']
X_hp_train, X_all_train, Y_train, df = load_auto(normalize=True)

# Training Parameters
learning_rates = [1, 1e-1, 1e-2, 1e-3, 1e-4]
costs_hp_models = []  # Costs for the Horsepower Only model for different learning rates
costs_all_models = []
num_iterations = 1000
iterations_list_hp = []
iterations_list_all = []

# Step 3: Train Models
for lr in learning_rates:
    
    # Model 1: Horsepower Only
    _, costs_hp, iters = train_linear_model(X_hp_train, Y_train, num_iterations, lr)
    costs_hp_models.append(costs_hp)
    iterations_list_hp.append(iters)
    
    iterations_list_all = []
    # Model 2: All Features
    _, costs_all, itres_all = train_linear_model(X_all_train, Y_train, num_iterations, lr)
    costs_all_models.append(costs_all)
    iterations_list_all.append(itres_all)


# Plot Costs vs Iterations for both models and all learning rates
plot_costs(costs_hp_models, iterations_list_hp[0], learning_rates, '(a) Model 1 (horsepower only)')
plot_costs(costs_all_models, iterations_list_all[0], learning_rates, '(b) Model 2 (all features)')

# Find the best learning rate for the horsepower only model
min_cost_hp = min([costs_hp_models[i][-1] for i in range(len(costs_hp_models))])
min_lr_hp = learning_rates[[costs_hp_models[i][-1] for i in range(len(costs_hp_models))].index(min_cost_hp)]
print(f'\nMinimum Cost for Horsepower Only Model: {min_cost_hp}')
print(f'Best Learning Rate for Horsepower Only Model: {min_lr_hp}')
### Best Learning Rate for Horsepower Only Model: 0.1


# Model 1: Horsepower Only (training for scatter plot visualization)
parameters_hp, costs_hp, _ = train_linear_model(X_hp_train, Y_train, num_iterations, min_lr_hp)
# Print the parameters for the horsepower only model
print(f'w = {parameters_hp["w"]}, b = {parameters_hp["b"]}\n')

# Visualization for Horsepower Model
plt.scatter(df['horsepower'], df['mpg'], color='blue', label='Actual Data')
hp_line = parameters_hp['w'][0,0] * df['horsepower'].values + parameters_hp['b']
plt.plot(df['horsepower'], hp_line, color='red', label='Regression Line')
plt.xlabel('Horsepower (Normalized)')
plt.ylabel('MPG')
plt.title(f'Linear Regression - Horsepower vs MPG (α={min_lr_hp})')
plt.legend()
# plt.show()
plt.savefig('plots/hp_vs_mpg_scatter_plot.png')

# Find the best learning rate for the all features model
# min_cost_all = min([costs_all_models[i][-1] for i in range(len(costs_all_models))])
# min_lr_all = learning_rates[[costs_all_models[i][-1] for i in range(len(costs_all_models))].index(min_cost_all)]
# Convert the list to a numpy array
costs_all_models_np = np.array([costs_all_models[i][-1] for i in range(len(costs_all_models))])

# Replace inf values with nan
costs_all_models_np[np.isinf(costs_all_models_np)] = np.nan

# Find the best learning rate for the all features model
min_cost_all = np.nanmin(costs_all_models_np)
min_lr_all = learning_rates[np.where(costs_all_models_np == min_cost_all)[0][0]]
print(f'\nMinimum Cost for All Features Model: {min_cost_all}')
print(f'Best Learning Rate for All Features Model: {min_lr_all}')
### Best Learning Rate for All Features Model: 0.1

# Train with the best learning rate for the all features model
parameters_all, costs_all, _ = train_linear_model(X_all_train, Y_train, num_iterations, min_lr_all)
# Print the parameters for the all features model
print(f'w = {parameters_all["w"]}, b = {parameters_all["b"]}\n')