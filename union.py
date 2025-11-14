"""
Union script: replica main_otra_prueba.py pero toma puntos y prioridades
generadas por main_flujo.py.

Comportamiento:
- Importa `main_flujo` (ejecuta su generación de puntos).
- Construye nodos donde el depósito es el índice 0 y los demás son los
  puntos seleccionados de `main_flujo`.
- Calcula costes euclídeos entre nodos.
- Obtiene prioridades desde `main_flujo.puntos_prioridad_adaptadas` o
  `main_flujo.puntos_prioridad` (busca la coincidencia por proximidad).
- Ejecuta el modelo de Gurobi similar a `main_otra_prueba.py`.

Nota: `main_flujo.py` ejecutará su código al importarse; si prefieres
exponer funciones en lugar de ejecutar en import, podemos refactorizar
`main_flujo.py` para devolver datos en funciones.
"""

import math
import numpy as np
try:
    from gurobipy import Model, GRB, quicksum
except Exception as e:
    raise ImportError("Gurobi no está disponible en este entorno: %s" % e)

import matplotlib.pyplot as plt

# Intentamos importar main_flujo que genera `puntos_sel` y `puntos_prioridad`.
import importlib
import main_flujo as mf
importlib.reload(mf)

# --- Helper: buscar prioridad dada una coordenada ---
def buscar_prioridad_por_coord(coord, dict_prioridades, tol=1e-6):
    # coord: (x,y)
    if dict_prioridades is None:
        return None
    # keys in dict_prioridades are tuples (x, y)
    # Intentar match exacto
    if tuple(coord) in dict_prioridades:
        return dict_prioridades[tuple(coord)]
    # Buscar por proximidad
    coords_keys = list(dict_prioridades.keys())
    if len(coords_keys) == 0:
        return None
    arr_keys = np.array(coords_keys)
    dists = np.linalg.norm(arr_keys - np.array(coord), axis=1)
    idx = int(np.argmin(dists))
    if dists[idx] <= 1e-3 or dists[idx] <= tol:
        return dict_prioridades[tuple(arr_keys[idx])]
    # fallback: devolver la prioridad del punto más cercano
    return dict_prioridades[tuple(arr_keys[idx])]


# --- Construir lista de coordenadas y prioridades a partir de main_flujo ---
# main_flujo define variables: puntos, puntos_sel, puntos_prioridad,
# puntos_prioridad_adaptadas, puntos_conocidos, etc.

# Elegir las prioridades adaptadas si existen, si no la original
prior_dict = None
if hasattr(mf, 'puntos_prioridad_adaptadas') and mf.puntos_prioridad_adaptadas:
    prior_dict = mf.puntos_prioridad_adaptadas
elif hasattr(mf, 'puntos_prioridad') and mf.puntos_prioridad:
    prior_dict = mf.puntos_prioridad

# Obtener puntos seleccionados (si existe), si no usar puntos totales
if hasattr(mf, 'puntos_sel') and mf.puntos_sel is not None:
    puntos_sel = np.array(mf.puntos_sel)
elif hasattr(mf, 'puntos') and mf.puntos is not None:
    puntos_sel = np.array(mf.puntos)
else:
    raise RuntimeError("main_flujo no contiene `puntos_sel` ni `puntos` después de importar")


def Gurobi_union(puntos_sel, prior_dict, depot=(0.0, 0.0)):
    import math
    from gurobipy import Model, GRB, quicksum

    # Construir coordenadas: índice 0 = depósito, 1..n = puntos seleccionados
    idx_to_coord = {0: depot}
    for i, p in enumerate(puntos_sel, start=1):
        idx_to_coord[i] = (float(p[0]), float(p[1]))

    coord_to_idx = {v: k for k, v in idx_to_coord.items()}
    V = list(idx_to_coord.keys())
    A = [(i, j) for i in V for j in V if i != j]

    def distancia(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    c = {(i, j): distancia(idx_to_coord[i], idx_to_coord[j]) for i, j in A}

    import random
    random.seed(42)
    p = {i: (0 if i == 0 else random.randint(1, 10)) for i in V}

    prioridad = {}
    for i in V:
        if i == 0:
            prioridad[i] = 0
        else:
            coord = idx_to_coord[i]
            # buscar_prioridad_por_coord debe coincidir con la manera que guardas las claves
            prio = prior_dict.get(tuple(np.round(coord,6)), None)
            prioridad[i] = float(prio if prio is not None else 1.0)

    # Parámetros y modelo (mantén los tuyos)
    T_max = 5; P_max = 80; I_max = 5; alpha = 1

    m = Model("ruta_ganancia_max_union")
    y = m.addVars(V, vtype=GRB.INTEGER, lb=0, ub=2, name="y")
    x = m.addVars(A, vtype=GRB.BINARY, name="x")
    peso_ij = m.addVars(A, vtype=GRB.CONTINUOUS, name="peso")

    for i in V:
        if i != 0:
            m.addConstr(y[i] <= 1, name=f"una_visita_{i}")
    m.addConstr(y[0] == 2, name="deposito_visitas")

    for i in V:
        if i != 0:
            m.addConstr(quicksum(x[i, j] for j in V if j != i) == y[i], name=f"salida_{i}")
            m.addConstr(quicksum(x[j, i] for j in V if j != i) == y[i], name=f"entrada_{i}")

    m.addConstr(quicksum(x[0, j] for j in V if j != 0) == 1, name="salida_0")
    m.addConstr(quicksum(x[j, 0] for j in V if j != 0) == 1, name="entrada_0")
    m.addConstr(quicksum(p[i]*y[i] for i in V) <= P_max, name="peso_max")
    m.addConstr(quicksum(y[i] for i in V if i != 0) <= I_max, name="items_max")

    for i, j in A:
        if i != 0 and i != j:
            m.addConstr(peso_ij[i, j] == quicksum(peso_ij[k, i]*x[k, i] for k in V if k != i) - p[i]*y[i])
        if i == 0:
            m.addConstr(peso_ij[0, j] == quicksum(p[k]*y[k] for k in V if k != 0) * x[0, j])

    m.setObjective(alpha * quicksum(prioridad[i]*y[i] for i in V) - quicksum(peso_ij[i, j]*x[i, j]*c[i, j] for i, j in A), GRB.MAXIMIZE)
    m.optimize()

    # nodos visitados (excepto depósito)
    visitados = [i for i in V if i != 0 and y[i].X > 0.5]
    puntos_visitados = [idx_to_coord[i] for i in visitados]

    # opcional: imprimir solución (como antes)



    # Extraer arcos usados
    arcos_usados = [(i, j) for i, j in A if x[i, j].X > 0.5]

    # Extraer puntos visitados
    puntos_visitados = [idx_to_coord[i] for i in range(1, len(idx_to_coord)) if u[i].X > 0]

    # RETORNAR la info necesaria
    return puntos_visitados, idx_to_coord, arcos_usados

