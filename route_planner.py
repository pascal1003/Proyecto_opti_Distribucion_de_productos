import math
import gurobipy as gp
from client_nodes import Client

class Planner:

  def __init__(self, max_items: int, max_weight: float=100.0):
    self.nodes = {0: Client(0, 0, 0, 0)}
    self.model = gp.Model(name="RoutePlanner")
    self.max_items = max_items
    self.max_weight = max_weight 

  @staticmethod
  def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

  def add_nodes(self, new_nodes: list[Client]):
    self.nodes = {0: Client(0, 0, 0, 0)}
    for node in new_nodes:
      self.nodes[node.id] = node

  @property
  def priorities(self):
    return {n.id: n.package.priority for n in self.nodes.values() if n.id != 0 and n.package}

  @property
  def adjusted_priorities(self):
    return {n.id: n.package.adjusted_priority for n in self.nodes.values() if n.id != 0 and n.package}

  @property
  def weights(self):
    return {n.id: n.package.weight for n in self.nodes.values() if n.id != 0 and n.package}

  @property
  def vertices(self):
    return list(self.nodes.keys())

  @property
  def edges(self):
    return [(i, j) for i in self.vertices for j in self.vertices if i != j]

  @property
  def costs(self):
    # Costs are normalized and then scaled in order to match the priority scale
    raw_costs = {(i.id, j.id): self.distance((i.x, i.y), (j.x, j.y))
           for i in self.nodes.values()
           for j in self.nodes.values()
           if i.id != j.id}
    max_dist = max(raw_costs.values()) if raw_costs else 1
    normalized = {k: v / max_dist for k, v in raw_costs.items()}
    SCALE = 50
    return {k: SCALE * v for k, v in normalized.items()}

  @staticmethod
  def _find_subtours(edges, visited_nodes):
    if not visited_nodes:
      return []

    successors = {i: j for i, j in edges}
    
    # Follow the tour that starts from the depot (main tour)
    main_tour_nodes = set()
    current_node = 0
    while True:
      next_node = successors.get(current_node)
      # Follow the tour until it breaks or ends
      if next_node is None or next_node == 0 or next_node in main_tour_nodes:
        break
      main_tour_nodes.add(next_node)
      current_node = next_node

    # Every node in the visited_nodes list that is not in the main tour is part of a subtour
    subtour_nodes = set(visited_nodes) - main_tour_nodes

    cycles = []
    while subtour_nodes:
      cycle = []
      start_node = next(iter(subtour_nodes)) # Grab any node from the subtour nodes set
      current_node = start_node

      # Follow the tour that contains this node
      while current_node in subtour_nodes:
        cycle.append(current_node)
        subtour_nodes.remove(current_node)
        current_node = successors.get(current_node)
      
      if len(cycle) > 1:
        # Add this cycle to the list of cycles
        cycles.append(cycle)

    return cycles

  def _subtour_elimination_callback(self, model, where):
    if where == gp.GRB.Callback.MIPSOL:
      # Get vertices and edges for the current solution
      v_vals = model.cbGetSolution(model._vars['v'])
      e_vals = model.cbGetSolution(model._vars['e'])
      
      selected_edges = [(i, j) for i, j in model._vars['e'].keys() if e_vals[i, j] > 0.5]
      visited_nodes = [i for i in model._vars['v'].keys() if v_vals[i] > 0.5 and i != 0]

      if not visited_nodes:
        return

      # F
      tours = self._find_subtours(selected_edges, visited_nodes)
      for tour in tours:
        if len(tour) > 1:
          model.cbLazy(
            gp.quicksum(model._vars['e'][i, j] for i in tour for j in tour if i != j) <= len(tour) - 1
          )

  def run_model(self):
    self.model.reset()
    V = self.vertices
    E = self.edges
    maxK = self.max_items

    # --- Variables ---
    v_var = self.model.addVars(V, vtype=gp.GRB.BINARY, name="v")
    e_var = self.model.addVars(E, vtype=gp.GRB.BINARY, name="e")
    weight_ij = self.model.addVars(E, vtype=gp.GRB.CONTINUOUS, name="weight")

    self.model._vars = {'v': v_var, 'e': e_var}
    
    # --- Restricciones (SIN MTZ) ---
    for v_idx in V:
      if v_idx == 0: continue
      self.model.addConstr(gp.quicksum(e_var[v_idx, j] for j in V if j != v_idx) == v_var[v_idx], f"exit_{v_idx}")
      self.model.addConstr(gp.quicksum(e_var[i, v_idx] for i in V if i != v_idx) == v_var[v_idx], f"entry_{v_idx}")
      
    self.model.addConstr(gp.quicksum(e_var[0, j] for j in V if j != 0) == v_var[0], "depot_out")
    self.model.addConstr(gp.quicksum(e_var[i, 0] for i in V if i != 0) == v_var[0], "depot_in")
    self.model.addConstr(v_var[0] == 1, "depot_visited")

    
    self.model.addConstr(gp.quicksum(v_var[i] for i in V if i != 0) <= maxK, "capacity")

    #Peso en arcos
    # 1. Cantidad de Items
    self.model.addConstr(gp.quicksum(v_var[i] for i in V if i != 0) <= maxK, "capacity_items")

    # 2. Peso Máximo
    self.model.addConstr(
        gp.quicksum(self.weights[i]*v_var[i] for i in V if i != 0) <= self.max_weight, 
        name="weight_capacity"
    )
    
    # --- Objetivo ---
    priority_term = gp.quicksum(100 * self.adjusted_priorities[j] * v_var[j] for j in V if j != 0)
    travel_cost = gp.quicksum(e_var[i, j] * self.costs[i, j] for i, j in E)
    self.model.setObjective(priority_term - travel_cost, gp.GRB.MAXIMIZE)

    # --- Parámetros y Optimización con Callback ---
    self.model.Params.LogToConsole = 1
    #self.model.Params.TimeLimit = 30
    self.model.Params.LazyConstraints = 1

    # Gurobi requires a plain Python function as callback (not a bound method),
    # so wrap the instance method in a local function closure and pass that.
    def _gurobi_callback(model, where):
      return self._subtour_elimination_callback(model, where)

    self.model.optimize(_gurobi_callback)

    # --- Recoger solución ---
    visited_nodes, traveled_edges = [], {}
    if self.model.Status in [gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT, gp.GRB.SUBOPTIMAL] and self.model.SolCount > 0:
      v_sol = self.model.getAttr('X', v_var)
      e_sol = self.model.getAttr('X', e_var)
      visited_nodes = [self.nodes[n] for n in V if n != 0 and v_sol[n] > 0.5]
      for (i, j) in E:
        if e_sol[i, j] > 0.5:
          xi, yi = self.nodes[i].x, self.nodes[i].y
          xj, yj = self.nodes[j].x, self.nodes[j].y
          traveled_edges[i, j] = [[xi, xj], [yi, yj]]

    return visited_nodes, traveled_edges, self.model.ObjVal