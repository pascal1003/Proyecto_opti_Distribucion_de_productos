from gurobipy import Model, GRB, quicksum
import csv
import matplotlib.pyplot as plt
import random


"""
# -------------------------
# Datos de ejemplo
# -------------------------
V = list(range(10))  # nodos (0 es depósito)
A = [(i, j) for i in V for j in V if i != j]  # arcos
p = {i: (i * 2 + 10) if i != 0 else 0 for i in V}  # pesos de productos, ejemplo: 10, 15, 20, ...
c = {(i, j): 1 for i, j in A}  # costos
t = {(i, j): 1 for i, j in A}  # tiempos de viaje
prioridad = {i: (60 - i) if i != 0 else 0 for i in V}  # prioridades, ejemplo: 9, 8, ..., 1, 0 para depósito
T_max = 5
P_max = 60
I_max = 5

alpha = 1

"""


# -------------------------
# Modelo
# -------------------------
m = Model("ruta_ganancia_max")

# Cargar nodos y arcos desde aristas.csv
def cargar_nodos_y_arcos(filename):
    nodos = set()
    arcos = []
    costos = {}
    with open(filename, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            source = (float(row['source_x']), float(row['source_y']))
            dest = (float(row['dest_x']), float(row['dest_y']))
            distancia = float(row['distance'])
            nodos.add(source)
            nodos.add(dest)
            arcos.append((source, dest))
            costos[(source, dest)] = distancia
    costos_retornables = []
    for n in nodos:
        c_aux = []
        for n2 in nodos:
            if n != n2:
                if (n, n2) in costos:
                    c_aux.append(costos[(n, n2)])
                else:
                    c_aux.append(float('inf'))  # o algún valor grande si no hay arco
            else:
                c_aux.append(0)
        costos_retornables.append(c_aux)
    return list(nodos), arcos, costos_retornables

_, _, c = cargar_nodos_y_arcos('aristas.csv')

print("Costos:", c)

V_coords, A_aristas, c_matrix = cargar_nodos_y_arcos('aristas.csv')

# Crea el mapeo de índice a coordenada
idx_to_coord = {idx: coord for idx, coord in enumerate(V_coords)}
coord_to_idx = {coord: idx for idx, coord in enumerate(V_coords)}

# Redefine V y A usando índices
V = list(range(len(V_coords)))
A = [(i, j) for i in V for j in V if i != j]

# Convierte la matriz de costos en un diccionario
c = {(i, j): c_matrix[i][j] for i in V for j in V}

# Ejemplo de pesos y prioridades (puedes ajustar según tu problema)
#p = {v: 0.1 for v in V}  # Peso de cada nodo
# reproducibilidad opcional
random.seed(42)

p = {v: (0 if v == 0 else random.randint(0, 10)) for v in V}
p = {i: (  10**i) if i != 0 else 0 for i in V} 
prioridad = {v: 100 for v in V}  # Prioridad de cada nodo
T_max = 5
P_max = 80
I_max = 5
alpha = 1

# Variables
y = m.addVars(V, vtype=GRB.INTEGER, lb=0, ub=2, name="y")
x = m.addVars(A, vtype=GRB.BINARY, name="x")
peso_ij = m.addVars(A, vtype=GRB.CONTINUOUS, name="peso")

# Restricciones

# Una visita por nodo, excepto depósito
for i in V:
    if i != 0:
        m.addConstr(y[i] <= 1, name=f"una_visita_{i}")

# Inicio y fin en depósito
m.addConstr(y[0] == 2, name="deposito_visitas")

# Salida y entrada de los nodos (excepto depósito)
for i in V:
    if i != 0:
        m.addConstr(quicksum(x[i, j] for j in V if j != i) == y[i], name=f"salida_{i}")
        m.addConstr(quicksum(x[j, i] for j in V if j != i) == y[i], name=f"entrada_{i}")

# Salida y entrada del depósito
m.addConstr(quicksum(x[0, j] for j in V if j != 0) == 1, name="salida_0")
m.addConstr(quicksum(x[j, 0] for j in V if j != 0) == 1, name="entrada_0")


# Peso máximo
m.addConstr(quicksum(p[i]*y[i] for i in V) <= P_max, name="peso_max")

# Items máximos (no contamos el depósito)
m.addConstr(quicksum(y[i] for i in V if i != 0) <= I_max, name="items_max")

#Peso en arcos
for i, j in A:
    if i != 0 and i!=j:
        m.addConstr(
            peso_ij[i, j] == quicksum(peso_ij[k, i]*x[k,i] for k in V if k != i) - p[i]*y[i]
        )
    if i == 0:
        m.addConstr(
            peso_ij[0, j] == quicksum(p[k]*y[k] for k in V if k != 0) * x[0, j]
        )


# Función objetivo
m.setObjective(
    alpha * quicksum(prioridad[i]*y[i] for i in V) - quicksum(peso_ij[i, j]*x[i, j]*c[i,j] for i, j in A) ,
    GRB.MAXIMIZE
)

# Optimizar
m.optimize()

print("Pesos:", p)
print("Prioridades:", prioridad)


# Mostrar resultados
for i in V:
    print(f"y[{i}] = {y[i].X}")
for i, j in A:
    if x[i, j].X > 0.5:
        print(f"x[{i},{j}] = {x[i,j].X}")
#print(peso_ij)
for i, j in A:
    if peso_ij[i, j].X > 0:
        print(f"peso_ij[{i},{j}] = {peso_ij[i,j].X*x[i,j].X}")


V_coords, A_aristas, _ = cargar_nodos_y_arcos('aristas.csv')
print =(prioridad)
# Plot nodes
plt.figure(figsize=(8, 8))
for idx, (x_coord, y_coord) in idx_to_coord.items():
    plt.plot(x_coord, y_coord, 'o', color='blue')
    plt.text(x_coord, y_coord, f'{idx}', fontsize=8, ha='right')



# Plot used arcs
for i, j in A:
    if x[i, j].X > 0.5:
        x0, y0 = idx_to_coord[i]
        x1, y1 = idx_to_coord[j]
        plt.plot([x0, x1], [y0, y1], 'r-', linewidth=2)

plt.title('Solución del modelo: nodos y aristas usadas')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()
