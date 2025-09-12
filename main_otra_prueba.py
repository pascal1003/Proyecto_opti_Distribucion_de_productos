from gurobipy import Model, GRB, quicksum

# -------------------------
# Datos de ejemplo
# -------------------------
V = [0, 1, 2, 3]  # nodos (0 es depósito)
A = [(i, j) for i in V for j in V if i != j]  # arcos
p = {1: 10, 2: 20, 3: 15, 0: 0}  # pesos de productos
c = {(i, j): 1 for i, j in A}  # costos
t = {(i, j): 1 for i, j in A}  # tiempos de viaje
T_max = 5
P_max = 40
I_max = 3

alpha = 1

# -------------------------
# Modelo
# -------------------------
m = Model("ruta_ganancia_max")

# Variables
y = m.addVars(V, vtype=GRB.INTEGER, lb=0, ub=2, name="y")
x = m.addVars(A, vtype=GRB.BINARY, name="x")

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


# Función objetivo
m.setObjective(
    alpha * quicksum(p[i]*y[i] for i in V) - quicksum(c[i, j]*x[i, j] for i, j in A) ,
    GRB.MAXIMIZE
)

# Optimizar
m.optimize()

# Mostrar resultados
for i in V:
    print(f"y[{i}] = {y[i].X}")
for i, j in A:
    if x[i, j].X > 0.5:
        print(f"x[{i},{j}] = {x[i,j].X}")