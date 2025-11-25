import math, gurobipy as gp
from client_nodes import Client

class Planner:
  def __init__(self, max_items: int):
    self.nodes = {0: Client(0, 0, 0, 0)}
    self.model = gp.Model(name="RoutePlanner")
    self.max_items = max_items

  @staticmethod
  def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

  def add_nodes(self, new_nodes:list[Client]):
    self.nodes = {0: Client(0, 0, 0, 0)} # Reset node dictionary
    for node in new_nodes:
      self.nodes[node.id] = node

  @property
  def weights(self):
    return {n.id: n.package.weight for n in self.nodes.values() if n.id != 0 and n.package}

  @property
  def priorities(self):
    return {n.id: n.package.adjusted_priority for n in self.nodes.values() if n.id != 0 and n.package}

  @property
  def vertices(self):
    return list(self.nodes.keys())

  @property
  def edges(self):
    # Build edges
    return [(i, j) for i in self.vertices for j in self.vertices if i != j]

  @property
  def costs(self):
    # distancias originales
    raw_costs = {(i.id, j.id): self.distance((i.x, i.y), (j.x, j.y))
                  for i in self.nodes.values()
                  for j in self.nodes.values()
                  if i.id != j.id}

    max_dist = max(raw_costs.values()) if raw_costs else 1
    normalized = {k: v / max_dist for k, v in raw_costs.items()}

    SCALE = 60

    return {k: SCALE * v for k, v in normalized.items()}

  def run_model(self):
    self.model.reset()  # Clean previous state
    # Aliases for sets
    V = self.vertices
    E = self.edges

    #####################################
    ## Variables
    # 
    v_var = self.model.addVars(V, vtype=gp.GRB.BINARY, name="v")                           # Node visited
    e_var = self.model.addVars(E, vtype=gp.GRB.BINARY, name="e")                           # Edge travelled
    u = self.model.addVars(V, vtype=gp.GRB.CONTINUOUS, lb=0, ub=self.max_items, name="u")  # Visit order

    #####################################
    ## Constraints
    #

    # 1) To visit a node, the entering edge and the exiting edge have to be travelled
    for v in V:
      if v == 0: continue
      self.model.addConstr(gp.quicksum(e_var[v, j] for j in V if j != v) == v_var[v], name=f"exit_{v}")
      self.model.addConstr(gp.quicksum(e_var[i, v] for i in V if i != v) == v_var[v], name=f"entry_{v}")

    # 2) Every route has to include the depot
    self.model.addConstr(gp.quicksum(e_var[0, j] for j in V if j != 0) == 1, name="depot_out")
    self.model.addConstr(gp.quicksum(e_var[i, 0] for i in V if i != 0) == 1, name="depot_in")

    # 3) Respect max item capacity
    self.model.addConstr(gp.quicksum(v_var[i] for i in V if i != 0) <= self.max_items, name="capacity")

    # 4) If node i is never visited, u[i] = 0, meaning it's position in the visiting order is 0 (invalid).
    #    The depot has u = 0 because it's the first node in the visiting chain
    self.model.addConstr(u[0] == 0, name="u_depot_zero")
    for i in V:
      if i == 0: continue
      # Node position in the visiting order can't be greater than the max number of visited nodes
      self.model.addConstr(u[i] <= self.max_items * v_var[i], name=f"u_upper_{i}")  
      # u[i] >= v[i], meaning that if node i is visited (v[i] = 1), u[i] has to be 1 or greater. If node is not visited, u[i] will be zero
      self.model.addConstr(u[i] >= v_var[i], name=f"u_lower_{i}")

    # 5) MTZ subtour elimination (aplica para i != j and i != 0 and j != 0)
    #  u_i - u_j + M * e_ij <= M - 1
    M = self.max_items
    for i, j in E:
      if i == 0 or j == 0:
        # Puedes incluir también las restricciones con depot si lo deseas; no es obligatorio
        continue
      self.model.addConstr(u[i] - u[j] + M * e_var[i, j] <= M - 1, name=f"mtz_{i}_{j}")

    #####################################
    ## Objective function
    #
    priority_term = gp.quicksum(100 * self.priorities[j] * v_var[j] for j in V if j != 0)
    travel_cost = gp.quicksum(e_var[i, j] * self.costs[i, j] for i, j in E)
    self.model.setObjective(priority_term - travel_cost, gp.GRB.MAXIMIZE)

    # Solve
    self.model.optimize()

    # Get solution and return
    visited_nodes = []
    if self.model.Status == gp.GRB.OPTIMAL or self.model.Status == gp.GRB.TIME_LIMIT or self.model.Status == gp.GRB.SUBOPTIMAL:
      visited_nodes = [self.nodes[n] for n in V if n != 0 and v_var[n].X > 0.5]
      traveled_edges = []
      for (i, j) in E:
        if e_var[i, j].X > 0.5:
          xi, yi = self.nodes[i].x, self.nodes[i].y
          xj, yj = self.nodes[j].x, self.nodes[j].y
          traveled_edges.append([[xi, xj], [yi, yj]])
    else:
      traveled_edges = []

    return visited_nodes, traveled_edges