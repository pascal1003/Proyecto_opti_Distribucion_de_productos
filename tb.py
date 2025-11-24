import random
import numpy as np, matplotlib.pyplot as plt, matplotlib.cm as cm, seaborn as sns
from client_nodes import Client
from route_planner import Planner
from sklearn import cluster

def tb(min_prio:int, max_prio:int, node_n:int, neighborhood_n:int, generator:str, iterations: int):
  #############################################
  ## Setup
  #
  random.seed(123)
  np.random.seed(123)

  #############################################
  ## Node map
  #
  node_list = []
  neighbourhood_centers = []
  known_nodes = []
  coords = None
  count = 0
  for i in range(neighborhood_n):
    # Choose a random coordinate to place the neighbourhood
    while (coords is None or coords in neighbourhood_centers):
      coords = tuple(random.randint(-50, 50) for _ in range(2))
    # Keep track of existing neighbourhoods
    neighbourhood_centers.append(coords)
    print(f"Neighbourhood {i} with center at {coords}")
    if (generator == "gauss"):
      # Create a gaussian cloud for each neighborhood
      cov = random.randint(30, 80)
      cov_matrix = [[cov, 0], [0, cov]]
      positions = np.random.multivariate_normal(mean=coords, cov=cov_matrix, size=node_n//neighborhood_n).tolist()
      # Generate a client at each position and assign a unique ID
      node_list += [Client(p[0], p[1], j, int(j*1e3)) for j, p in enumerate(positions, start=count + 1)] 
      # Keep track of how many clients have been generated (so IDs stay unique)
      count += node_n//neighborhood_n
    
    elif (generator == "normal"):
      ### TODO: implement uniform random maps
      pass

  x_positions = [n.x for n in node_list]
  y_positions = [n.y for n in node_list]
  min_x, max_x = min(x_positions), max(x_positions)
  min_y, max_y = min(y_positions), max(y_positions)

  #############################################
  ## Plot the map
  #
  map_fig, map_ax = plt.subplots()
  map_ax.scatter([n.x for n in node_list], [n.y for n in node_list])
  map_ax.scatter([0], [0], marker="x", color="black")
  map_ax.set_xlim(min_x-10, max_x+10)
  map_ax.set_ylim(min_y-10, max_y+10)
  map_ax.set_title("Client map")

  for iteration in range(iterations):
    #############################################
    ## Generate packages for the nodes
    #
    active_nodes = random.choices(node_list, k=random.randint(1, int(node_n * 0.05)))
    for node in active_nodes:
      # A node is "active" when the associated client has generated a package
      node.generate_package(min_prio, max_prio)
      print(f"Node {node.id} generated package with priority {node.package.priority}")
      if node not in known_nodes:
        # Clients that have generated packages are known to the planner
        known_nodes.append(node)

    #############################################
    ## Plot known nodes
    #
    known_fig, known_ax = plt.subplots()
    known_ax.set_title("Known clients")
    known_ax.set_xlim(min_x-10, max_x+10)
    known_ax.set_ylim(min_y-10, max_y+10)
    known_ax.scatter([0], [0], marker="x", color="black")
    sns.scatterplot(x=[n.x for n in known_nodes], y=[n.y for n in known_nodes], edgecolor="black", hue=[n.package.priority for n in known_nodes], axes=known_ax, palette="crest")
    legend = known_ax.get_legend()
    if legend is not None:
      legend.set_title("Package priority")

    #############################################
    ## Generate clusters
    #
    node_pos = np.array([[n.x, n.y] for n in known_nodes])
    clusters = cluster.DBSCAN(eps=15, min_samples=1).fit(node_pos)
    labels = clusters.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print("Estimated number of clusters: %d" % n_clusters_)
    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[clusters.core_sample_indices_] = True

    colors = [cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    cluster_fig, clusters_ax = plt.subplots()
    clusters_ax.set_xlim(min_x-10, max_x+10)
    clusters_ax.set_ylim(min_y-10, max_y+10)
    clusters_ax.scatter([0], [0], marker="x", color="black")
    for k, col in zip(unique_labels, colors):
        if k == -1: continue  # Ignore noise
        class_member_mask = labels == k
        xy = node_pos[class_member_mask]
        clusters_ax.plot(xy[:, 0], xy[:, 1], "o", markerfacecolor=tuple(col), markeredgecolor="k", label=k)

        # Adjust priority for nodes in the cluster
        member_nodes = []
        member_count = 0
        for j, n in enumerate(class_member_mask):
          if n:
            member_nodes.append(known_nodes[j])
            member_count += 1
        for node in member_nodes:
          if node.package.priority > 0:
            node.package.adjusted_priority = node.package.priority / member_count
    
    clusters_ax.set_title(f"Identified clusters: {n_clusters_}")
    
    #############################################
    ## Plan the route
    #
    planner = Planner(15, 100.0)
    planner.add_nodes([n for n in known_nodes if n.package and n.package.priority > 0])
    visited_nodes, traveled_edges = planner.run_model()
    print("Nodes visited:")
    count = 0
    for node in visited_nodes:
      if node.id == 0 or not node.package: continue
      print(f"\t{node.id} -> priority {node.package.priority}")
      count += 1
    print(f"Total: {count}")

    #############################################
    ## Plot the route
    #
    route_fig, route_ax = plt.subplots()
    route_ax.set_title(f"Optimal route for iteration {iteration + 1}")
    route_ax.scatter([0], [0], marker="x", color="black")
    route_ax.set_xlim(min_x-10, max_x+10)
    route_ax.set_ylim(min_y-10, max_y+10)
    print([(n.x, n.y) for n in visited_nodes])
    for edge in traveled_edges:
      print(edge)
      route_ax.plot(*edge, 'r-')
    sns.scatterplot(x=[n.x for n in known_nodes], y=[n.y for n in known_nodes], edgecolor="black", hue=[n.package.priority for n in known_nodes], axes=route_ax, palette="crest")

    #############################################
    ## Mark delivered packages as having zero priority
    #
    for node in visited_nodes:
      if node.id == 0 or not node.package: continue
      node.package.priority = 0
    
    max_age = 0
    for node in node_list:
      if node.package and node.package.priority > 0:
        node.package.age += 1
        if node.package.age > max_age:
          max_age = node.package.age 
    print(f"Oldest package is {max_age} iterations old")

    plt.show()

if __name__ == "__main__":
  tb(1, 3, 500, 2, "gauss", 5)