import random
import numpy as np, matplotlib.pyplot as plt, seaborn as sns
from client_nodes import Client
from route_planner import Planner
from sklearn import cluster

def tb(min_prio:int, max_prio:int, node_n:int, neighborhood_n:int, generator:str):
  #############################################
  ## Setup
  #
  random.seed(123)
  np.random.seed(123)

  #############################################
  ## Global variables
  #
  known_nodes = []

  #############################################
  ## Node map
  #
  node_list = []
  neighbourhood_centers = []
  coords = None
  count = 0
  for i in range(neighborhood_n):
    while (coords is None or coords in neighbourhood_centers):
      coords = tuple(random.randint(-50, 50) for _ in range(2))
    neighbourhood_centers.append(coords)
    print(f"Neighbourhood {i} with center at {coords}")
    if (generator == "gauss"):
      # Create a gaussian cloud for each neighborhood
      cov = random.randint(30, 80)
      cov_matrix = [[cov, 0], [0, cov]]
      positions = np.random.multivariate_normal(mean=coords, cov=cov_matrix, size=node_n//neighborhood_n).tolist()
      node_list += [Client(p[0], p[1], j, int(j*1e3)) for j, p in enumerate(positions, start=count + 1)] 
      count += node_n//neighborhood_n
    
    elif (generator == "normal"):
      pass
  
  #############################################
  ## Generate packets for the nodes
  #
  active_nodes = random.choices(node_list, k=random.randint(1, node_n//3))
  for node in active_nodes:
    node.generate_packet(min_prio, max_prio)
    print(f"Node {node.id} generated packet with priority {node.packet.priority}")
    if node not in known_nodes:
      known_nodes.append(node)

  
  #############################################
  ## Plot the map
  #
  map_fig, map_ax = plt.subplots()
  map_ax.scatter([n.x for n in node_list], [n.y for n in node_list])

  #############################################
  ## Plot known nodes
  #
  known_fig, known_ax = plt.subplots()
  sns.scatterplot(x=[n.x for n in known_nodes], y=[n.y for n in known_nodes], hue=[n.packet.priority for n in known_nodes], axes=known_ax)
  legend = known_ax.get_legend()
  if legend is not None:
    legend.set_title("Priority")

  #############################################
  ## Generate clusters
  #
  node_pos = np.array([[n.x, n.y] for n in known_nodes])
  clusters = cluster.DBSCAN(eps=15, min_samples=1).fit(node_pos)
  labels = clusters.labels_

  # Number of clusters in labels, ignoring noise if present.
  n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
  n_noise_ = list(labels).count(-1)
  print("Estimated number of clusters: %d" % n_clusters_)
  print("Estimated number of noise points: %d" % n_noise_)
  unique_labels = set(labels)
  core_samples_mask = np.zeros_like(labels, dtype=bool)
  core_samples_mask[clusters.core_sample_indices_] = True

  colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
  cluster_fig, clusters_ax = plt.subplots()
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
        if node.packet.priority > 0:
          node.packet.priority /= member_count
  
  clusters_ax.set_title(f"Estimated number of clusters: {n_clusters_}")
  
  #############################################
  ## Plan the route
  #
  planner = Planner(50)
  planner.add_nodes(known_nodes)
  visited_nodes, traveled_edges = planner.run_model()
  print("Nodes visited:")
  count = 0
  for node in visited_nodes:
    if node.id == 0: continue
    print(f"\t{node.id} -> priority {node.packet.priority}")
    count += 1
  print(f"Total: {count}")

  #############################################
  ## Plot the route
  #
  route_fig, route_ax = plt.subplots()
  route_ax.scatter([n.x for n in visited_nodes], [n.y for n in visited_nodes], color="blue")
  print([(n.x, n.y) for n in visited_nodes])
  for edge in traveled_edges:
    print(edge)
    route_ax.plot(*edge, 'r-')

  plt.show()

if __name__ == "__main__":
  tb(1, 3, 500, 2, "gauss")