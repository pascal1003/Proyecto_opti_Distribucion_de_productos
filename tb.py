import random
import numpy as np, matplotlib.pyplot as plt, matplotlib.cm as cm, seaborn as sns
from pathlib import Path
from sklearn import cluster
from client_nodes import Client
from route_planner import Planner

def tb(min_prio:int, max_prio:int, node_n:int, neighborhood_n:int, generator:str, iterations: int, prio_adjust: bool):
  #############################################
  ## Setup
  #
  random.seed(123)
  np.random.seed(123)
  Path(f"results/{neighborhood_n}_{node_n}_{generator}{'_prioadjust' if prio_adjust else ''}").mkdir(parents=True, exist_ok=True)
  log_file = open(f"results/{neighborhood_n}_{node_n}_{generator}{'_prioadjust' if prio_adjust else ''}/log.txt", "w")
  results_file = open(f"results/{neighborhood_n}_{node_n}_{generator}{'_prioadjust' if prio_adjust else ''}/results.csv", "w")
  log_file.write(f"Running {iterations} iterations with {node_n} nodes divided into {neighborhood_n} neighborhoods.\n")
  results_file.write("iteration,objective_val,profit,nodes_visited,oldest\n")

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
  map_fig.savefig(f"results/{neighborhood_n}_{node_n}_{generator}{'_prioadjust' if prio_adjust else ''}/map.pdf")

  for iteration in range(1, iterations + 1):
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
    sns.scatterplot(x=[n.x for n in known_nodes], y=[n.y for n in known_nodes], edgecolor="black", hue=[n.package.priority for n in known_nodes], hue_norm=(0, max_prio), axes=known_ax)
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

        if prio_adjust:
          # Adjust priority for nodes in the cluster
          member_nodes = []
          member_count = 0
          active_members = 0
          for j, n in enumerate(class_member_mask):
            if n:
              member_nodes.append(known_nodes[j])
              member_count += 1
              if known_nodes[j].package and known_nodes[j].package.priority > 0:
                active_members += 1
          for node in member_nodes:
            if node.package.priority > 0:
              node.package.adjusted_priority = node.package.priority * active_members / member_count
    
    clusters_ax.set_title(f"Identified clusters: {n_clusters_}")
    
    #############################################
    ## Plan the route
    #
    print(f"Delivering {np.sum([n.package.weight for n in known_nodes if n.package and n.package.priority > 0])} kg")
    planner = Planner(15)
    planner.add_nodes([n for n in known_nodes if n.package and n.package.priority > 0])
    visited_nodes, traveled_edges, objective_value = planner.run_model()
    print("Nodes visited:")
    count = 0
    for node in visited_nodes:
      if node.id == 0 or not node.package: continue
      print(f"\t{node.id} -> priority {node.package.priority}")
      count += 1
    print(f"Total: {count}")
    print(f"Delivered {np.sum([n.package.weight for n in visited_nodes if n.package])} kg")

    #############################################
    ## Plot the route
    #
    route_fig, route_ax = plt.subplots()
    route_ax.set_title(f"Optimal route for iteration {iteration}")
    route_ax.scatter([0], [0], marker="x", color="black", zorder=float("inf"))
    route_ax.set_xlim(min_x-10, max_x+10)
    route_ax.set_ylim(min_y-10, max_y+10)
    for (i,j), edge in traveled_edges.items():
      route_ax.plot(*edge, 'r-')
    sns.scatterplot(x=[n.x for n in known_nodes], y=[n.y for n in known_nodes], edgecolor="black", hue=[n.package.priority for n in known_nodes], hue_norm=(0, max_prio), axes=route_ax, zorder=float("inf"))

    #############################################
    ## Calculate profit for this route
    profit = 100 * np.sum([n.package.priority for n in visited_nodes if n.package]) - np.sum([planner.costs[e] for e in traveled_edges.keys()])

    #############################################
    ## Mark delivered packages as having zero priority
    #
    for node in visited_nodes:
      if node.id == 0 or not node.package: continue
      node.package.priority = 0
    
    oldest_node = None
    for node in node_list:
      if node.package and node.package.priority > 0:
        node.package.age += 1
        if oldest_node is None or node.package.age > oldest_node.package.age:
          oldest_node = node
    if oldest_node:
      print(f"Oldest package is {oldest_node.package.age} iterations old")

    #############################################
    ## Log information about this iteration
    #
    log_file.write(f"\nIteration {iteration}:\n")
    log_file.write(f"\tObjective function value:{objective_value}\n")
    log_file.write(f"\tProfit value:{profit}\n")
    log_file.write(f"\tNodes visited: {len(visited_nodes)}\n")
    if oldest_node:
      log_file.write(f"\tOldest undelivered package: {oldest_node.id} ({oldest_node.package.age} iterations)\n")
    results_file.write(f"{iteration},{objective_value},{profit},{len(visited_nodes)},{oldest_node.package.age if oldest_node else "nan"}\n")

    route_fig.savefig(f"results/{neighborhood_n}_{node_n}_{generator}{'_prioadjust' if prio_adjust else ''}/route_{iteration}.pdf")
    known_fig.savefig(f"results/{neighborhood_n}_{node_n}_{generator}{'_prioadjust' if prio_adjust else ''}/known_{iteration}.pdf")
    cluster_fig.savefig(f"results/{neighborhood_n}_{node_n}_{generator}{'_prioadjust' if prio_adjust else ''}/cluster_{iteration}.pdf")
    #plt.show()
  
  #############################################
  ## Close files
  #
  log_file.close()
  results_file.close()

if __name__ == "__main__":
  tb(1, 3, 500, 2, "gauss", 5, True)
  tb(1, 3, 500, 2, "gauss", 5, False)
  tb(1, 3, 500, 3, "gauss", 5, True)
  tb(1, 3, 500, 3, "gauss", 5, False)
  tb(1, 3, 1000, 10, "gauss", 20, True)
  tb(1, 3, 1000, 10, "gauss", 20, False)