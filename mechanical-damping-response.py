# -*- coding: utf-8 -*-
"""


Created on Fri Apr 30 07:46:22 2021

@author: et140552
"""
# %% Importing necessary packages
import os
import math
import numpy as np
import scipy as sp
from scipy.signal import find_peaks
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import *
from tkinter import filedialog
plt.style.use('default')
plt.close('all')

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

# %% Input sample size and test setup 

# Assign findpeaks and sample sizes
window_size = 1000      # unit in milliseconds
number_of_peaks = 15    # How many waves you want to sample
first_peak = 0          # Start sampling from nth peak


# Number of files dictate number of rotation angle increment in radians
rotation_angle_increment = 360 / (2*math.pi*len(files))

# %% Creating empty lists 

df_table_of_results=[]
df_filenames = []
df_logarithmic_decrement = []
df_period =[]
df_damping_ratio =[]
df_damping_frequency =[]
df_natural_frequency =[]

# %% Looping 

# Read and loop for each selected files
for file in files:

    # Get base filename    
    filename = os.path.basename(file)
    
    df_filenames.append(filename)
    
    # Read only gauge values
    df_gauge_values = pd.read_csv(file,
                                  names = ["Gauge1", "Gauge2", "Gauge3"],
                                  skiprows = 29, # Skip first 29 rows of csv file
                                  usecols = [0, 1, 2] # Use first 3 columns of csv file
                                  )
    
    df_gauge_values["GaugeRad"] = np.sqrt(df_gauge_values['Gauge1']**2 +
                                          df_gauge_values['Gauge2']**2 +
                                          df_gauge_values['Gauge3']**2)
    
    # Designate the maximum Gauge Rad as the start index + 20ms 
    # And end 1 second after index start as the end index
    # [index_start:index_end] will be used as the window
    
    index_start = np.argmax(df_gauge_values["GaugeRad"]) + 20
    
    index_end = index_start + window_size
    
    df_gauge_values_window = df_gauge_values.loc[index_start:index_end]
    
    # Reset index to match the findpeaks function
    df_gauge_values_window=df_gauge_values_window.reset_index()


    # %% Find indices of peaks on each gauge values
    # Take only the 2nd to 32nd peaks and filter width 4ms
    
    gauge_peak_indices = [find_peaks(df_gauge_values_window["Gauge1"], width = 2)[0][first_peak:first_peak+number_of_peaks],
                          find_peaks(df_gauge_values_window["Gauge2"], width = 2)[0][first_peak:first_peak+number_of_peaks],
                          find_peaks(df_gauge_values_window["Gauge3"], width = 2)[0][first_peak:first_peak+number_of_peaks],
                          find_peaks(df_gauge_values_window["GaugeRad"], width = 2)[0][2*first_peak:2*(first_peak+number_of_peaks)],
                          ]

    # Skip every other element to account for RSSQ
    gauge_peak_indices[3] = gauge_peak_indices[3][::2]

    # Calculate logarithmic decrement
    """
    Important: 
        If the gauge output y_0 is equal to y_n the natural log of (y_0/y_n) is 0.
        This will prompt error ZeroDivisionError
        This can be observed if the gauge output does NOT decrease over time ie NO damping present.
        
    Solution:
        Change number of peaks to be used in analysis
    """
    
    logarithmic_decrement = [(number_of_peaks)**-1 * math.log(df_gauge_values_window["Gauge1"][gauge_peak_indices[0][0]] / df_gauge_values_window["Gauge1"][gauge_peak_indices[0][-1]]),
                             (number_of_peaks)**-1 * math.log(df_gauge_values_window["Gauge2"][gauge_peak_indices[1][0]] / df_gauge_values_window["Gauge2"][gauge_peak_indices[1][-1]]),
                             (number_of_peaks)**-1 * math.log(df_gauge_values_window["Gauge3"][gauge_peak_indices[2][0]] / df_gauge_values_window["Gauge3"][gauge_peak_indices[2][-1]]),
                             (number_of_peaks)**-1 * math.log(df_gauge_values_window["GaugeRad"][gauge_peak_indices[3][0]] / df_gauge_values_window["GaugeRad"][gauge_peak_indices[3][-1]])]
    
    df_logarithmic_decrement.append(logarithmic_decrement)

    # Calculate damping ratio
    damping_ratio = [1/(np.sqrt((1 + (2*math.pi / logarithmic_decrement[0])**2))),
                     1/(np.sqrt((1 + (2*math.pi / logarithmic_decrement[1])**2))),
                     1/(np.sqrt((1 + (2*math.pi / logarithmic_decrement[2])**2))),
                     1/(np.sqrt((1 + (2*math.pi / logarithmic_decrement[3])**2)))]

    df_damping_ratio.append(damping_ratio)


    # Calculate period 
    period = [np.mean(gauge_peak_indices[0][2:number_of_peaks] - gauge_peak_indices[0][1:number_of_peaks-1]),
              np.mean(gauge_peak_indices[1][2:number_of_peaks] - gauge_peak_indices[1][1:number_of_peaks-1]),
              np.mean(gauge_peak_indices[2][2:number_of_peaks] - gauge_peak_indices[2][1:number_of_peaks-1]),
              np.mean(gauge_peak_indices[3][2:number_of_peaks] - gauge_peak_indices[3][1:number_of_peaks-1])]
      
    df_period.append(period)
            
    # Calculate damping frequency
    damping_frequency = [1/period[0],
                         1/period[1],
                         1/period[2],
                         1/period[3]]

    df_damping_frequency.append(damping_frequency)

    # Calculate natural frequency
    
    natural_frequency = [damping_frequency[0] / np.sqrt((1 - damping_ratio[0])**2),
                         damping_frequency[1] / np.sqrt((1 - damping_ratio[1])**2),
                         damping_frequency[2] / np.sqrt((1 - damping_ratio[2])**2),
                         damping_frequency[3] / np.sqrt((1 - damping_ratio[3])**2)]
    
    df_natural_frequency.append(natural_frequency)    

    #%% Plot Charts
    fig, (ax1, ax2) = plt.subplots(2, sharex= True)
    fig.suptitle(os.path.basename(file))

    ax1.plot(df_gauge_values_window["Gauge1"], label = "Gauge 1")
    x = gauge_peak_indices[0]
    y = [df_gauge_values_window["Gauge1"][j] for j in gauge_peak_indices[0]]
    ax1.scatter(x , y, marker="x")
    ax1.legend()
    
    ax1.plot(df_gauge_values_window["Gauge2"], label = "Gauge 2")
    x = gauge_peak_indices[1]
    y = [df_gauge_values_window["Gauge2"][j] for j in gauge_peak_indices[1]]
    ax1.scatter(x , y, marker="x")

    ax1.plot(df_gauge_values_window["Gauge3"], label = "Gauge 3")
    x = gauge_peak_indices[2]
    y = [df_gauge_values_window["Gauge3"][j] for j in gauge_peak_indices[2]]
    ax1.scatter(x , y, marker="x")
    ax1.legend()

    
    ax2.plot(df_gauge_values_window["GaugeRad"], label = "RSSQ Gauge")
    x = gauge_peak_indices[3]
    y = [df_gauge_values_window["GaugeRad"][j] for j in gauge_peak_indices[3]]
    ax2.scatter(x , y, marker="x")
    ax2.legend()
    
    ax1.set(ylabel='Gauge Output')
    ax2.set(xlabel='Time [ms]', ylabel = 'Gauge Rad Output')

      

    """
    Using package plotly
    
    # Create new chart
    fig = go.Figure()
    
    # Plot Gauge 1 with markers on peaks
    fig.add_trace(go.Scatter(
        y=df_gauge_values_window["Gauge1"],
        mode='lines',
        name='Gauge 1',))
    
    fig.add_trace(go.Scatter(
        x=gauge_peak_indices[0],
        y=[df_gauge_values_window["Gauge1"][j] for j in gauge_peak_indices[0]],
        mode='markers',
        marker=dict(
            size=8,
            symbol='cross'),
        name='Gauge 1 Detected Peaks'))
    
    # Plot Gauge 2 with markers on peaks
    fig.add_trace(go.Scatter(
        y=df_gauge_values_window["Gauge2"],
        mode='lines',
        name='Gauge 2'))
    
    fig.add_trace(go.Scatter(
        x=gauge_peak_indices[1],
        y=[df_gauge_values_window["Gauge2"][j] for j in gauge_peak_indices[1]],
        mode='markers',
        marker=dict(
            size=8,
            symbol='cross'),
        name='Gauge 2 Detected Peaks'))

    # Plot Gauge 3 with markers on peaks
    fig.add_trace(go.Scatter(
        y=df_gauge_values_window["Gauge3"],
        mode='lines',
        name='Gauge 3'))
    
    fig.add_trace(go.Scatter(
        x=gauge_peak_indices[2],
        y=[df_gauge_values_window["Gauge3"][j] for j in gauge_peak_indices[2]],
        mode='markers',
        marker=dict(
            size=8,
            symbol='cross'),
        name='Gauge 3 Detected Peaks'))
    
    fig.update_xaxes(
        title_text = "Time [ms]",
        )
    fig.update_yaxes(
        title_text = "Gauge Value",
        )
    fig.show()
    """

#%% Collate results
df_logarithmic_decrement = pd.DataFrame.from_records(df_logarithmic_decrement,
                                                     columns = ['logdec1',
                                                                'logdec2',
                                                                'logdec3',
                                                                'logdecRad']
                                                     )

df_period = pd.DataFrame.from_records(df_period,
                                      columns = ['period1',
                                                 'period2',
                                                 'period3',
                                                 'periodRad']
                                      )

df_damping_ratio = pd.DataFrame.from_records(df_damping_ratio,
                                             columns = ['dampingratio1',
                                                        'dampingratio2',
                                                        'dampingratio3',
                                                        'dampingratioRad']
                                             )

df_damping_frequency = pd.DataFrame.from_records(df_damping_frequency,
                                                 columns = ['dampingfreq1',
                                                            'dampingfreq2',
                                                            'dampingfreq3',
                                                            'dampingfreqRad']
                                                 )

df_natural_frequency = pd.DataFrame.from_records(df_natural_frequency,
                                                 columns = ['naturalfreq1',
                                                            'naturalfreq2',
                                                            'naturalfreq3',
                                                            'naturalfreqRad']
                                                 )

df_table_of_results = pd.concat([df_logarithmic_decrement,
                                 df_period,
                                 df_damping_ratio,
                                 df_damping_frequency,
                                 df_natural_frequency],
                                axis = 1)

df_table_of_results.insert(0, 'Records', df_filenames)

#%% Polar Plot of Logarithimic Decrement

fig = plt.figure()
theta = np.linspace(0.0, 2 * math.pi, len(files), endpoint=False)
theta = theta.tolist()
theta1 = theta[0]
theta.append(theta1)

r = df_table_of_results['logdecRad']
r1 = r.append(pd.Series(r[0]))
plt.polar(theta, r1)

plt.show()

