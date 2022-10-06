
from tkinter import N
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
import matplotlib
import matplotlib.pylab as plt
import seaborn as sns
import pulp

import warnings
warnings.filterwarnings("ignore")

#VARIABLES Y PARAMETROS

customer = []
x_coord = []
y_coord = []
d = []                  #demanda
a = []                  #ready time
b = []                  #due time
t = []                  #service time 

file = open('in1.txt', 'r')
Lines = file.readlines()
line_count = 0
for line in Lines:
    if line_count == 0: 
        n_vehicle  = input()
        q = input()             #capacidad de cada vehiculo 
    else:
        temp = line.split("\t")
        customer.append(temp[0])
        x_coord.append(temp[1])
        y_coord.append(temp[2])
        d.append[temp[3]]
        a.append(temp[4])
        b.append(temp[5])
        t.append(temp[6])
    line_count += 1
file.close()

n_customer = len(customer)
n_points = n_customer + 2

#CALCULO DE 
df = pd.DataFrame({
    'x': x_coord,
    'y': y_coord,
})
df.iloc[0]['x'] = n_points
df.iloc[0]['y'] = n_points
dist = pd.DataFrame(distance_matrix(df[['x', 'y']].values, df[['x', 'y']].values), index=df.index, columns=df.index).values

fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(distances, ax=ax, cmap='Blues', annot=True, fmt='.0f', cbar=True, cbar_kws={"shrink": .3}, linewidths=.1)
plt.title('distance matrix')
plt.show()

#SETEAR PROBLEMA
problem = pulp.LpProblem('vrptw', pulp.LpMinimize)

#SETEAR VARIABLES
x = pulp.LpVariable.dicts('x', ((i, j) for i in range(n_points) for j in range(n_points)), lowBound=0, upBound=1, cat='Binary')     #aristas seleccionadas para el recorrido
vehicles = pulp.LpVariable('n_vehicle', lowBound=0, upBound=n_vehicle, cat='Integer')                                               #cantidad de vehiculos a usar
u = pulp.LpVariable.dicts('u', (i for i in range(n_points)), lowBound = 1, upBound = q, cat = 'Integer')                            #capacidad usada    
h = [0] * vehicles
#el limite de tiempo cuenta como variable conciderando la minimizacion de vehiculos?

#SETEAR FO
problem += pulp.lpSum([dist[i][j] * x[i, j] for i in range(n_points) for j in range(n_points)])

#SETEAR RESTRICCIONES 
for i in range(n_points):
    problem += x[i, i] == 0 # impide loops 

#Grafo completo 
for i in range(1, n_points):
    problem += pulp.lpSum(x[j, i] for j in range(n_points)) == 1 # Constraint (2)
    problem += pulp.lpSum(x[i, j] for j in range(n_points)) == 1 # Contraint (3)

#vuelve la misma cantidad de vehiculos que salen desde el sumidero    
problem += pulp.lpSum(x[i, 0] for i in range(n_points)) == n_vehicle # Constraint (4)
problem += pulp.lpSum(x[0, i] for i in range(n_points)) == n_vehicle # Constraint (5)

#SETEAR MTZ
#?????
for i in range(n_points):
    for j in range(n_points):
        if i != j and (i != 0 and j != 0):
            problem += u[i] - u[j] + d[i] <= n_points * (1 - x[i, j]) # Constraint (8)

#Demanda acumulada menor que capacidad de  vehiculo 
for i in range(n_points):
    if i!=0:
        problem += 1 <= u[i] <= q - d[i] # Constraint (9)

#RESTRICCIONES DE TIEMPO
for k in range(vehicles):
    for i in range(n_points):
        for j in range(n_points):

            if(h[k] < a[i]):
                h[k] = a[i]
            problem += x[i,j] * (h[k] + t[i] + - a[j]) <= 0     # Constraint de secuencia de intervalo 
            problem += a[i] <= h[k] + t[i] <= b[i]                # Constraint de tamaño de deposito en intervalo
            h[k] += x[i,j] * t[i]
            

    

#RESOLVER 
solver = pulp.getSolver('PULP_CBC_CMD', timeLimit=120) #120 seg.

# solve problem
status = problem.solve(solver)

# output status, value of objective function
status, pulp.LpStatus[status], pulp.value(problem.objective)


