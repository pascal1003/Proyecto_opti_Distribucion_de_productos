import random

class Package:
  def __init__(self, priority:int):
    self.priority = priority
    self.adjusted_priority = priority
    self.age = 0
    self.weight = random.uniform(1.0, 10.0)  # weight in kg

class Client:
  def __init__(self, x:float, y:float, id:int, seed:int):
    self.x = x
    self.y = y
    self.id = id
    self.package = None
    self.rand = random.Random(seed)
  
  def generate_package(self, min_prio:int, max_prio:int)->None:
    self.package = Package(self.rand.randint(min_prio, max_prio))