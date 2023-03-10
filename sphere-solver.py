# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 08:27:44 2022

@author: et140552
"""

# Import necessary packages
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import scipy
from scipy import optimize
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'browser'
import tkinter as tk
from tkinter import Tk, filedialog, Button
plt.style.use('ggplot')

# %% Locate directory of data to be imported via tkinter dialog box
root = tk.Tk()

root.filename = filedialog.askopenfilenames(initialdir = "/",
                                            title = "Select LTR file to be analysed",
                                            )

files = root.filename

# Create button to exit tkinter
button_exit = Button(root,  
                     text = "Exit", 
                     command = root.destroy)

button_exit.grid(column = 1, row = 1)

root.mainloop()

# Terminate program if nothing is selected
if files == (""):
    raise SystemExit("Nothing selected")

#%% Import raw 3Dsphere data, 
# Note: No headers (date, username, TCR... etc)

raw_data = pd.read_csv(files[0],
                       names = ["X_POS", "Y_POS", "Z_POS", "RAD", "ANGLE", "ELEVATION", "FEEDRATE"],
                       sep = '[\\\,]', #double \\ removes special property of \ resulting in one \ as delimiter. \s means space. 
                       engine = 'python'
                       )

# Calculate the root sum sqaure (i.e. radius)
raw_data["RSS"] = (raw_data.X_POS**2 + raw_data.Y_POS**2 + raw_data.Z_POS**2)**0.5

#%% Statistics

# Grouped by all angle elevation
# Note all units calculated in millimeters [mm]
mean_data = raw_data.groupby(['ELEVATION', 'ANGLE'], as_index=False).agg(np.mean)
max_data = raw_data.groupby(['ELEVATION', 'ANGLE'], as_index=False).agg(np.max)
min_data = raw_data.groupby(['ELEVATION', 'ANGLE'], as_index=False).agg(np.min)
std_data = raw_data.groupby(['ELEVATION', 'ANGLE'], as_index=False).agg(np.std)

span_data = max_data - min_data
two_std_data = 2*std_data


#%% Fitting algorithms

"""
Fit function uses the equation of a sphere to calculate the magnitude, 
the root squared of xyz input data subtraced from XYZ error (origin)
"""

def fit(x, y, z, X, Y, Z): 
    
    magnitude = math.sqrt((x-X)**2 + (y-Y)**2 + (z-Z)**2) 
    
    return magnitude


"""
Mid Fit function calculates electronic stylus ball radius, 
3D radius center and 3D radius error 
M is input data, M2 is initial and iterated guess
"""

def form_fit(M, M2): 
    
    # Unpack the 1D arrays of X, Y and Z points
    x, y, z = M
    X, Y, Z = M2
    centred_radius = np.zeros(x.shape)
    centred_error = np.zeros(x.shape)
           
    # Iterate input data and find x, y, z centred positions
    for i in range(0, x.size):            
        
        # Use fit function on input data and initial guess data
        temp = fit(x[i], y[i], z[i], X, Y, Z) 
        
        # Calculate the electronic ball radius
        ball_rad_new = fit(x[0], y[0], z[0], X, Y, Z) - input_data.RSS[0]            
        
        # Calculate the 3D radius centre
        temp = temp - ball_rad_new             
        
        # Fill array with centred data
        centred_radius[i] = temp
    
    # Iterate centred data and find 3D radius error
    for i in range(0, x.size):            
        
        # Calculate 3D radius error
        temp = centred_radius[i] - np.mean(centred_radius)           
        
        # Fill array with normalised centred data
        centred_error[i] = temp
    
    # Find sum square error of every 3D rad error
    sum_square_3d_error = sum(map(lambda x: x*x, centred_error)) 
    
    return sum_square_3d_error, centred_radius, centred_error


 
"""
Sphere Fit function initiates minimisation of xyz errors
"""

def sphere_fit():
    
    # Use initial guess of xyz_error = [0,0,0]
    initial_guess = [0, 0, 0] 
    
    # Update the initial_guess for corrected xyz_error
    def solution(args):             

        # Leaves guess data missing to be fit by initial_guess
        return form_fit(xyz_data, args)[0] 

    # Minimises sum square error in form_fit
    results = scipy.optimize.least_squares(solution, initial_guess) 
    
    return results


#%% Assign input data and apply functions

# Note: raw/mean/max/min/span/std datasets can be used
input_data = mean_data

xyz_data = np.vstack((input_data['X_POS'],
                      input_data['Y_POS'],
                      input_data['Z_POS']))

# Apply the sphere fitting algorithm for each xyz point in input_data
# Note xyz_error is the origin of the hemisphere
optimise = sphere_fit()
xyz_error = optimise.x
                    
# Update input dataframe with fitted data
form = form_fit(xyz_data, xyz_error)[1:3]

input_data["X Centred"] = input_data['X_POS'] - xyz_error[0]
input_data["Y Centred"] = input_data['Y_POS'] - xyz_error[1]
input_data["Z Centred"] = input_data['Z_POS'] - xyz_error[2]
input_data["3D Radius Centred"] = (input_data["X Centred"]**2 + input_data["Y Centred"]**2 + input_data["Z Centred"]**2)**0.5
input_data["3D Radius Error"] = form[1]

#%%


fig = go.Figure()

fig.add_trace(go.Scatter3d(x = input_data['X_POS'],
                           y = input_data['Y_POS'],
                           z = input_data['Z_POS'],
                           name = 'Raw Data',
                           mode = 'markers',
                           marker = dict(size = [1],
                                         )
                           ))

fig.add_trace(go.Scatter3d(x = input_data['X Centred'],
                            y = input_data['Y Centred'],
                            z = input_data['Z Centred'],
                            name = 'Optimised Data',
                            mode = 'markers',
                            marker = dict(size = [1],
                                          )
                            ))

fig.show()
