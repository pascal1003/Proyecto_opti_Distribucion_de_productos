import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
#--------- Generar mapa total ----------
# Número total de puntos
n_puntos = 500

# Mitad para cada distribución
n1 = n_puntos // 2
n2 = n_puntos - n1

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
plt.scatter(puntos[:,0], puntos[:,1], c='blue', alpha=0.6, label='Puntos')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('100 puntos distribuidos según dos Gaussianas')
plt.legend()
plt.grid(True)
plt.show()


#--------- Seleccionar primeros pedidos ----------

# Seleccionar puntos aleatorios
n_seleccion = 10
indices_sel = np.random.choice(len(puntos), n_seleccion, replace=False)
puntos_sel = puntos[indices_sel]
puntos_conocidos = puntos_sel.copy()

# Asignar prioridades aleatorias y asociarlas a cada punto
prior_min = 1
prior_max = 10
prioridades = np.random.randint(prior_min, prior_max + 1, size=n_seleccion)

# Crear diccionario { (x, y): prioridad }
puntos_prioridad = {tuple(p): int(prio) for p, prio in zip(puntos_sel, prioridades)}

#-------ciclo principal del program----------
for i in range(3):
    # Graficar
    plt.figure(figsize=(7,7))
    plt.scatter(puntos[:,0], puntos[:,1], c='lightgray', alpha=0.4, label='Todos los puntos')

    # Puntos conocidos que no estan seleccionados
    puntos_sin_prioridad = [
        p for p in puntos_conocidos
        if tuple(np.round(p, 2)) not in puntos_prioridad.keys()
    ]

    if puntos_sin_prioridad:
        puntos_sin_prioridad = np.array(puntos_sin_prioridad)
        plt.scatter(
            puntos_sin_prioridad[:,0],
            puntos_sin_prioridad[:,1],
            c='black',
            s=100,
            label='Puntos conocidos'
        )

    # Colorear los puntos con prioridad según su valor
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
    n_iteraciones = 10


    # Realizar clustering
    dbscan = DBSCAN(eps=15, min_samples=1)
    labels = dbscan.fit_predict(puntos_conocidos)

    #Graficar
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
    # Calcular N_cluster y N_cluster_en_lista
    clusters_unicos = np.unique(labels)
    puntos_prioridad_adaptadas = {}
    print("largo seleccionados = ", len(puntos_sel))
    print("largo concidos = ", len(puntos_conocidos))
    for cluster_id in clusters_unicos:
        # puntos del cluster actual
        puntos_cluster = puntos_conocidos[labels == cluster_id]
        N_cluster = len(puntos_cluster)
        print("N_cluster = ", N_cluster)
        # puntos del cluster que están en puntos_sel
        N_cluster_en_lista = sum(
            tuple(p) in [tuple(ps) for ps in puntos_sel]
            for p in puntos_cluster
        )
        print("N_cluster_en_lista = ", N_cluster_en_lista)
        # calcular prioridad adaptada para cada punto conocido
        for p in puntos_cluster:
            punto_tupla = tuple(p)
            if punto_tupla in puntos_prioridad:  # si es un punto conocido
                prioridad_original = puntos_prioridad[punto_tupla]
                prioridad_adaptada = prioridad_original * N_cluster_en_lista / N_cluster
                puntos_prioridad_adaptadas[punto_tupla] = prioridad_adaptada

    # --- Graficar ---
    plt.figure(figsize=(7,7))
    plt.scatter(puntos[:,0], puntos[:,1], c='lightgray', alpha=0.4, label='Todos los puntos')

    # Graficar puntos conocidos con prioridad adaptada
    scatter = plt.scatter(
        [p[0] for p in puntos_prioridad_adaptadas.keys()],
        [p[1] for p in puntos_prioridad_adaptadas.keys()],
        c=list(puntos_prioridad_adaptadas.values()),
        cmap='plasma',
        s=120,
        edgecolor='black',
        label='Puntos con prioridad adaptada'
    )

    plt.colorbar(scatter, label='Prioridad adaptada')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Prioridad adaptada por densidad de clúster')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("Puntos seleccionados con prioridad adaptada:")
    for (x, y), prioridad in puntos_prioridad_adaptadas.items():
        print(f"({x:.2f}, {y:.2f}) -> prioridad adaptada {prioridad:.2f}")


    #---------------cambiar esta seccion por las pociciones que saca Gurobi----------
    #------------------------------------------------------------------------------------------
    # Ordenar y eliminar los 3 puntos con mayor prioridad -----
    # Ordenar claves del diccionario por prioridad descendente
    puntos_ordenados = sorted(puntos_prioridad_adaptadas.items(), key=lambda x: x[1], reverse=True)
    puntos_a_eliminar = [p for p, _ in puntos_ordenados[:3]]
    # Eliminar del diccionario
    for p in puntos_a_eliminar:
        puntos_prioridad.pop(p, None)
    # Eliminar esos puntos de los puntos seleccionados
    puntos_sel = np.array([p for p in puntos_sel if tuple(p) not in puntos_a_eliminar])
    #---------------Fin de la zona a cambiar por guirobi----------------
    #------------------------------------------------------------------------------------------


    #-----Aumentar prioridad basal a los puntos restantes -----
    aumento_prior = 1 # aumentamos en un 100% la prioridad basal
    for p in puntos_prioridad:
        puntos_prioridad[p] = round(puntos_prioridad[p] * (1 + aumento_prior), 2)

    # Mostrar resultados
    print("Puntos restantes con prioridad actualizada:")
    for (x, y), prio in puntos_prioridad.items():
        print(f"({x:.2f}, {y:.2f}) -> prioridad {prio}")

    #------- Agregar nuevos puntos ------
    max_nuevas = 5
    n_nuevas = np.random.randint(0, max_nuevas + 1)
    indices_nuevos = np.random.choice(len(puntos), n_nuevas, replace=False)
    puntos_nuevos = puntos[indices_nuevos]

    #Agregar a listas existentes
    puntos_sel = np.vstack((puntos_sel, puntos_nuevos))
    puntos_conocidos = np.vstack((puntos_conocidos, puntos_nuevos))

    #Asignar prioridades a los nuevos puntos
    prioridades_nuevas = np.random.randint(prior_min, prior_max + 1, size=n_nuevas)

    #Agregar al diccionario puntos_prioridad
    for p, prio in zip(puntos_nuevos, prioridades_nuevas):
        puntos_prioridad[tuple(p)] = int(prio)

    # Mostrar resultados
    print(f"Se agregaron {n_nuevas} nuevos puntos:")
    for (x, y), prio in zip(puntos_nuevos, prioridades_nuevas):
        print(f"({x:.2f}, {y:.2f}) -> prioridad {prio}")
    print("Puntos restantes con prioridad actualizada:")
    for (x, y), prio in puntos_prioridad.items():
        print(f"({x:.2f}, {y:.2f}) -> prioridad {prio}")