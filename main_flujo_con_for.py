import math, numpy as np, matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from gurobipy import Model, GRB, quicksum

#--------- Parámetros ---------
N_PUNTOS = 500
N_SELECCION = 10
PRIO_MIN = 1
PRIO_MAX = 10
TIEMPO_MAX = 50
PESO_MAX = 800
ITEMS_MAX = 50
ALPHA = 10
MAX_NUEVAS = 5

#--------- Generar mapa total ----------

# Mitad para cada distribución
n1 = N_PUNTOS // 2
n2 = N_PUNTOS - n1

# Parámetros de las dos gaussianas
media1 = [50, 80]
media2 = [80, 50]
cov1 = [[50, 0], [0, 50]]  # matriz de covarianza (dispersión)
cov2 = [[100, 0], [0, 100]]
# Generar puntos
puntos1 = np.random.multivariate_normal(media1, cov1, n1)
puntos2 = np.random.multivariate_normal(media2, cov2, n2)

# Unir los puntos
puntos = np.vstack((puntos1, puntos2))

# Graficar
plt.figure(figsize=(6,6))
plt.scatter(puntos[:,0], puntos[:,1], c='blue', ALPHA=0.6, label='Puntos')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('{N_PUNTOS} puntos distribuidos según dos Gaussianas')
plt.legend()
plt.grid(True)
plt.show()


#--------- Seleccionar primeros pedidos ----------

# Seleccionar puntos aleatorios
indices_sel = np.random.choice(len(puntos), N_SELECCION, replace=False)
puntos_sel = puntos[indices_sel]
puntos_conocidos = puntos_sel.copy()

# Asignar prioridades aleatorias y asociarlas a cada punto
prioridades = np.random.randint(PRIO_MIN, PRIO_MAX + 1, size=N_SELECCION)

# Crear diccionario { (x, y): prioridad }
puntos_prioridad = {tuple(p): int(prio) for p, prio in zip(puntos_sel, prioridades)}

#-------ciclo principal del programa----------
for i in range(5):
    # Graficar
    plt.figure(figsize=(7,7))
    plt.scatter(puntos[:,0], puntos[:,1], c='lightgray', ALPHA=0.4, label='Todos los puntos')

    # Puntos conocidos que no estan seleccionados
    puntos_sin_prioridad = [
        p for p in puntos_conocidos
        if tuple(np.round(p, 2)) not in puntos_prioridad.keys()
    ]

    if len(puntos_sin_prioridad) > 0:
        puntos_sin_prioridad = np.array(puntos_sin_prioridad)
        plt.scatter(
            puntos_sin_prioridad[:,0],
            puntos_sin_prioridad[:,1],
            c='black',
            s=100,
            label='Puntos conocidos'
        )

    # Colorear los puntos con prioridad según su valor
    if len(puntos_prioridad) > 0:
        scatter = plt.scatter(
            [p[0] for p in puntos_prioridad.keys()],
            [p[1] for p in puntos_prioridad.keys()],
            c=list(puntos_prioridad.values()),
            cmap='plasma',
            s=120,
            edgecolor='black',
            label='Puntos seleccionados'
        )
        plt.colorbar(scatter, label='Prioridad')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Puntos seleccionados con prioridad (mapa de calor)')
    plt.legend()
    plt.grid(True)
    plt.show()

    #--------- Algoritmo de optimizacion ------------
    #--------- Clustering ------------
    dbscan = DBSCAN(eps=15, min_samples=1)
    # Si no hay puntos conocidos evitar fit
    if len(puntos_conocidos) == 0:
        print("No hay puntos conocidos; ciclo finalizado.")
        break

    labels = dbscan.fit_predict(puntos_conocidos)

    # Graficar clustering
    plt.figure(figsize=(6,6))
    plt.scatter(puntos_conocidos[:,0], puntos_conocidos[:,1], c=labels, cmap='tab10', s=120, edgecolor='k')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Clustering con DBSCAN sobre puntos_conocidos')
    plt.grid(True)
    plt.show()

    # Mostrar los labels
    print("Etiquetas de clúster asignadas:", labels)

    #---------- Nueva prioridad ---------------
    clusters_unicos = np.unique(labels)
    puntos_prioridad_adaptadas = {}
    print("largo seleccionados = ", len(puntos_sel))
    print("largo concidos = ", len(puntos_conocidos))
    # Preparar lista de puntos_sel como tuplas para comparación
    puntos_sel_tuplas = [tuple(np.round(ps,6)) for ps in puntos_sel] if len(puntos_sel) > 0 else []

    for cluster_id in clusters_unicos:
        puntos_cluster = puntos_conocidos[labels == cluster_id]
        N_cluster = len(puntos_cluster)
        print("N_cluster = ", N_cluster)
        # puntos del cluster que están en puntos_sel
        N_cluster_en_lista = sum(tuple(np.round(p,6)) in puntos_sel_tuplas for p in puntos_cluster)
        print("N_cluster_en_lista = ", N_cluster_en_lista)
        # calcular prioridad adaptada para cada punto conocido
        for p in puntos_cluster:
            punto_tupla = tuple(np.round(p,6))
            if punto_tupla in puntos_prioridad:  # si es un punto conocido (las claves de puntos_prioridad deben estar coherentes)
                prioridad_original = puntos_prioridad[punto_tupla]
                # evitar división por cero
                prioridad_adaptada = prioridad_original * (N_cluster_en_lista / N_cluster) if N_cluster > 0 else prioridad_original
                puntos_prioridad_adaptadas[punto_tupla] = prioridad_adaptada



    print("Puntos seleccionados con prioridad adaptada:")
    for (x, y), prioridad in puntos_prioridad_adaptadas.items():
        print(f"({x:.2f}, {y:.2f}) -> prioridad adaptada {prioridad:.2f}")

    # ----------- Preparar diccionario de prioridades para Gurobi -------------
    # Usamos claves redondeadas para asegurar coincidencia con coordenadas que Gurobi devolverá
    prior_dict_for_gurobi = {tuple(np.round(k,6)): v for k, v in puntos_prioridad.items()}

    # ----------- Ejecutar Gurobi y obtener puntos visitados -------------
    puntos_visitados = []
    if len(puntos_sel) > 0:
        # Asumo que tu Gurobi_union ahora tiene la firma: Gurobi_union(puntos_sel, prior_dict)
        depot = (0, 0)

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
                prio = prior_dict_for_gurobi.get(tuple(np.round(coord,6)), None)
                prioridad[i] = float(prio if prio is not None else 1.0)

        # Modelo
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
        m.addConstr(quicksum(p[i]*y[i] for i in V) <= PESO_MAX, name="peso_max")
        m.addConstr(quicksum(y[i] for i in V if i != 0) <= ITEMS_MAX, name="items_max")

        for i, j in A:
            if i != 0 and i != j:
                m.addConstr(peso_ij[i, j] == quicksum(peso_ij[k, i]*x[k, i] for k in V if k != i) - p[i]*y[i])
            if i == 0:
                m.addConstr(peso_ij[0, j] == quicksum(p[k]*y[k] for k in V if k != 0) * x[0, j])

        m.setObjective(ALPHA * quicksum(10*prioridad[i]*y[i] for i in V) - quicksum(peso_ij[i, j]*x[i, j]*c[i, j] for i, j in A), GRB.MAXIMIZE)
        m.optimize()

        # nodos visitados (excepto depósito)
        visitados = [i for i in V if i != 0 and y[i].X > 0.5]
        puntos_visitados = [idx_to_coord[i] for i in visitados]

        # opcional: imprimir solución (como antes)



    else:
        print("No hay puntos_sel para optimizar con Gurobi.")

    print("\nPuntos visitados por Gurobi:")
    for p in puntos_visitados:
        print(p)

    # Convertir a tuplas redondeadas
    puntos_visitados_tupla = [tuple(np.round(p, 6)) for p in puntos_visitados]

    # ----------- Eliminar puntos visitados ---------------
    if len(puntos_visitados_tupla) > 0:
        if len(puntos_sel) > 0:
            puntos_sel = np.array([p for p in puntos_sel if tuple(np.round(p, 6)) not in puntos_visitados_tupla])
        if len(puntos_conocidos) > 0:
            puntos_conocidos = np.array([p for p in puntos_conocidos if tuple(np.round(p, 6)) not in puntos_visitados_tupla])

        for p in puntos_visitados_tupla:
            puntos_prioridad.pop(p, None)

    #-----Aumentar prioridad basal a los puntos restantes -----
    aumento_prior = 1  # aumentamos en un 100% la prioridad basal
    for p in list(puntos_prioridad.keys()):
        puntos_prioridad[p] = round(puntos_prioridad[p] * (1 + aumento_prior), 2)
    # --------- Graficar solución de Gurobi ----------
    plt.figure(figsize=(8, 8))

    # Nodos
    for idx, (x_coord, y_coord) in idx_to_coord.items():
        plt.plot(x_coord, y_coord, 'o', color='blue')
        plt.text(x_coord, y_coord, f'{idx}', fontsize=8, ha='right')

    # Arcos
    arcos_usados = [(i, j) for i, j in A if x[i, j].X > 0.5]
    for i, j in arcos_usados:
        x0, y0 = idx_to_coord[i]
        x1, y1 = idx_to_coord[j]
        plt.plot([x0, x1], [y0, y1], 'r-', linewidth=2)

    plt.title('Solución del modelo: nodos y aristas usadas')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()

    # Mostrar resultados
    print("Puntos restantes con prioridad actualizada:")
    for (x, y), prio in puntos_prioridad.items():
        print(f"({x:.2f}, {y:.2f}) -> prioridad {prio}")

    #------- Agregar nuevos puntos ------
    n_nuevas = np.random.randint(0, MAX_NUEVAS + 1)
    if n_nuevas > 0:
        indices_nuevos = np.random.choice(len(puntos), n_nuevas, replace=False)
        puntos_nuevos = puntos[indices_nuevos]

        #Agregar a listas existentes (controlar si están vacías)
        if len(puntos_sel) == 0:
            puntos_sel = np.array(puntos_nuevos)
        else:
            puntos_sel = np.vstack((puntos_sel, puntos_nuevos))

        if len(puntos_conocidos) == 0:
            puntos_conocidos = np.array(puntos_nuevos)
        else:
            puntos_conocidos = np.vstack((puntos_conocidos, puntos_nuevos))

        #Asignar prioridades a los nuevos puntos
        prioridades_nuevas = np.random.randint(PRIO_MIN, PRIO_MAX + 1, size=n_nuevas)

        #Agregar al diccionario puntos_prioridad
        for p, prio in zip(puntos_nuevos, prioridades_nuevas):
            puntos_prioridad[tuple(np.round(p,6))] = int(prio)

        # Mostrar resultados
        print(f"Se agregaron {n_nuevas} nuevos puntos:")
        for (x, y), prio in zip(puntos_nuevos, prioridades_nuevas):
            print(f"({x:.2f}, {y:.2f}) -> prioridad {prio}")

    print("Puntos restantes con prioridad actualizada:")
    for (x, y), prio in puntos_prioridad.items():
        print(f"({x:.2f}, {y:.2f}) -> prioridad {prio}")
