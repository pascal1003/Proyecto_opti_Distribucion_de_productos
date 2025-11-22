import random

class Packet:
  def __init__(self, priority:int):
    self.priority = priority


class Client:
  def __init__(self, x:float, y:float, id:int, seed:int):
    self.x = x
    self.y = y
    self.id = id
    self.packet = None
    self.rand = random.Random(seed)
  
  def generate_packet(self, min_prio:int, max_prio:int)->None:
    self.packet = Packet(self.rand.randint(min_prio, max_prio))