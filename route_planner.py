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
    for node in new_nodes:
      self.nodes[node.id] = node

  @property
  def priorities(self):
    return {n.id: n.package.adjusted_priority for n in self.nodes.values() if n.id != 0}

  @property
  def vertices(self):
    return self.nodes.keys()

  @property
  def edges(self):
    return [(i, j) for i in self.vertices for j in self.vertices if i != j and (i < j or j == 0) ]

  @property
  def costs(self):
    return {(i.id,j.id): self.distance((i.x, i.y), (j.x, j.y)) for i in self.nodes.values() for j in self.nodes.values() if i != j }

  def run_model(self):
    v_var = self.model.addVars(self.vertices, vtype=gp.GRB.INTEGER, lb=0, ub=2, name="vertices")
    e_var = self.model.addVars(self.edges, vtype=gp.GRB.BINARY, name="edges")

    for v in self.vertices:
      if v == 0: continue
      self.model.addConstr(v_var[v] <= 1, name=f"Only visit once")
      self.model.addConstr(gp.quicksum(e_var[v, j] for j in self.vertices if j != v and (v,j) in self.edges) == v_var[v], name=f"exit")
      self.model.addConstr(gp.quicksum(e_var[j, v] for j in self.vertices if j != v and (j,v) in self.edges) == v_var[v], name=f"entry")
    
    self.model.addConstr(v_var[0] == 2, name="Visit deposit twice")
    self.model.addConstr(gp.quicksum(e_var[0, j] for j in self.vertices if j != 0) == 1, name="reachable from deposit")
    self.model.addConstr(gp.quicksum(e_var[j, 0] for j in self.vertices if j != 0) == 1, name="can reach deposit")
    self.model.addConstr(gp.quicksum(v_var[i] for i in self.vertices if i != 0) <= self.max_items, name="Item capacity")

    self.model.setObjective(100 * gp.quicksum(self.priorities[j] * v_var[j] for j in self.vertices if j != 0) - gp.quicksum(e_var[i, j] * self.costs[i, j] for i, j in self.edges), gp.GRB.MAXIMIZE)
    self.model.optimize()

    visited_nodes = [n for n in self.nodes.values() if v_var[n.id].X > 0]
    traveled_edges = []
    for i,j in [(m, n) for m,n in self.edges if e_var[m, n].X > 0]:
      traveled_edges.append([[self.nodes[i].x, self.nodes[j].x], [self.nodes[i].y, self.nodes[j].y]])


    self.model.reset()

    return visited_nodes, traveled_edges