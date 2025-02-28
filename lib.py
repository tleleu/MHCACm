#   Timothee Leleu, Sam Reifenstein
#   Non-Equilibrium Dynamics of Hybrid Continuous-Discrete Ground-State Sampling
#   ICLR2025

import numpy as np
import pandas as pd
import os
from datetime import datetime
from collections import Counter
from itertools import product

def create_timestamped_folder(solvertype):
    # Get the current date and time
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Construct the folder name
    folder_name = f"{current_time}_{solvertype}"

    # Check if the folder already exists
    if not os.path.exists(folder_name):
        # Create the folder if it doesn't exist
        os.makedirs(folder_name)
        print(f"Folder created: {folder_name}")
    else:
        print(f"Folder already exists: {folder_name}")

    return folder_name


def save_to_file(folder_name, file_name, f_eval, evalist, param_out):
    # Construct the file name
    file_path = os.path.join(folder_name, file_name)

    # Write the data to the file
    with open(file_path, 'w') as file:
        file.write(f"{f_eval}\n")
        file.write(" ".join(map(str, np.exp(param_out))) + "\n")
        file.write(" ".join(map(str, evalist)))
        

    print(f"Data saved to file: {file_path}")


def read_file(folder_name, file_name):
    file_path = os.path.join(folder_name, file_name)
        
    p0=[]
    params=[]
    pvec=[]
    
    if os.path.exists(file_path):

        with open(file_path, 'r') as file:
            p0 = float(file.readline().strip())
            params = list(map(float, file.readline().strip().split()))
            pvec = list(map(float, file.readline().strip().split()))

    return p0, params, pvec

def count_vector_occurrences(L, L0):

    # Create a Counter for the vectors in L
    counter = Counter(tuple(vector) for vector in L)
    
    # Count the occurrences of each vector in L0
    occurrences = [counter[tuple(vector)] for vector in L0]
    
    return occurrences