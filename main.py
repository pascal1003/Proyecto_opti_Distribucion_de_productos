import pulp

# Datos de ejemplo
V = [0,1,2,3]  # Vértices
A = [(i,j) for i in V for j in V if i != j]  # Arcos

# Parámetros
p = {0:0, 1:5, 2:10, 3:8}  # Ganancias
c = {(0,1):2, (0,2):3, (0,3):4,  # Costos
     (1,0):2, (1,2):1, (1,3):2,
     (2,0):3, (2,1):1, (2,3):3,
     (3,0):4, (3,1):2, (3,2):3}
for (i,j) in A:
    if (i,j) not in c:
        c[(i,j)] = c[(j,i)]

t = c.copy()  # Tiempos / costos
Tmax = 10
Pmax = 20
Imax = 3
alpha = 1.0
model = pulp.LpProblem("Ruta_Ganancia_Minima_Tiempo_Max", pulp.LpMaximize)

# Variables
y = pulp.LpVariable.dicts("y", V, cat='Binary')
x = pulp.LpVariable.dicts("x", A, cat='Binary')

# Objetivo
model += alpha * pulp.lpSum(p[i] * y[i] for i in V) - pulp.lpSum(c[(i,j)] * x[(i,j)] for (i,j) in A)

# Restricciones
# Flujo out
for i in V:
    model += pulp.lpSum(x[(i,j)] for j in V if (i,j) in A) == y[i]

# Flujo in
for i in V:
    model += pulp.lpSum(x[(j,i)] for j in V if (j,i) in A) == y[i]

# Tiempo max
model += pulp.lpSum(t[(i,j)] * x[(i,j)] for (i,j) in A) <= Tmax

# Peso max
model += pulp.lpSum(p[i] * y[i] for i in V) <= Pmax

# Items max
model += pulp.lpSum(y[i] for i in V) <= Imax + 1  # +1 por depósito

# Forzar visita al depósito
model += y[0] == 1

# ChatGPT para problema de subtour: Ciclo cerrado en el grafo que no incluye el vértice 0 y 
# no está conectado al resto
# Eliminación subtours: Para pequeño n, uso MTZ (Miller-Tucker-Zemlin) en vez de exponencial
u = pulp.LpVariable.dicts("u", [i for i in V if i != 0], lowBound=0, upBound=len(V)-1, cat='Integer')
for i in V:
    if i == 0: continue
    for j in V:
        if j == 0 or i == j: continue
        model += u[i] - u[j] + (len(V)-1) * x[(i,j)] <= len(V)-2

# Resolver
status = model.solve(pulp.PULP_CBC_CMD(msg=0))

# Output
print("Status:", pulp.LpStatus[status])
print("Objetivo:", pulp.value(model.objective))
print("Visitas:", {i: pulp.value(y[i]) for i in V})
print("Arcos:", {(i,j): pulp.value(x[(i,j)]) for (i,j) in A if pulp.value(x[(i,j)]) > 0.5})