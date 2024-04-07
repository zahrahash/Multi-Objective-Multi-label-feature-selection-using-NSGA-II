from NSGAII import NSGA2
import moo as mtp
from LoadData import read_datas_from_Mulan
import numpy as np

# Load data from the MULAN database
DatasetName = input("Please enter the dataset name: ")
Data = read_datas_from_Mulan(DatasetName) #includes: bibtex - scene - emotions - enron - genbase - medical - yeast - birds - Corel5k 
num_features = Data[0].shape[1]

# Problem Definition
problem = {
    'cost_function': lambda solution: mtp.feature_selection_objectives(solution, Data),
    'n_var': Data[0].shape[1] # Number of Decision Variables
}

# Initialize Algorithm
alg = NSGA2(
    max_iter = 100, #Maximum Number of Iterations
    pop_size = 30, # Population Size
    p_crossover = 0.7, # Crossover Percentage
    p_mutation = 1/num_features,
    mu = 0.05,
    verbose = True,
)

# Solve the Problem
results = alg.run(problem)
pop = results['pop']
F = results['F']
