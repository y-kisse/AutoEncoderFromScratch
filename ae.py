import random
import math

LEARNING_RATE = 0.1
TRAINING_EPOCHS = 1000
NUM_OF_TRAINING_DATA = 100

def create_random_sequence(N):
  return [random.randint(0, 1) * 1.0 for _ in range(N)]

def sigmoid(x):
  return 1/(1 + math.pow(math.e, -x))

class Node:
  def __init__(self, y):
    self.v = 0
    self.y = y
    self.d = 0

    return

  def add(self, x):
    self.v = self.v + x
    self.y = sigmoid(self.v)
    return

  def getV(self):
    return self.v

  def getY(self):
    return self.y

  def addBackDelta(self, x):
    self.d = self.d + x
    return

  def getHiddenDelta(self):
    return sigmoid(self.v) * self.d

  def getOutDelta(self, ans):
    return (self.getY() - ans) * (1 - sigmoid(self.getV())) * sigmoid(self.getV())

if __name__ == '__main__':

  # user input.
  print('Number of input units: ', end='')
  N = int(input())

  print('Number of intermidiate units: ', end='')
  M = int(input())

  # Initialize network weights.
  weights = [[[random.random() for _ in range(M)] for _ in range(N)], [[random.random() for _ in range(N)] for _ in range(M)]]

  # make 100 Training Data.
  training_dataset = [create_random_sequence(N) for _ in range(NUM_OF_TRAINING_DATA)]

  # training

  for cycle in range(TRAINING_EPOCHS):
    E = 0.0
    for data in training_dataset:
      # Initialize nodes.
      nodes = [[Node(y) for y in data], [Node(0) for _ in range(M)], [Node(0) for _ in range(N)]]
      
      # input -> intermediate
      for i in range(N):
        for j in range(M):
          nodes[1][j].add(nodes[0][i].getY() * weights[0][i][j])

      # intermediate -> output
      for i in range(M):
        for j in range(N):
          nodes[2][j].add(nodes[1][i].getY() * weights[1][i][j])

      # calc Error

      for i in range(N):
        E = E + (math.pow(nodes[0][i].getY() - nodes[2][i].getY(), 2) / 2)

    # Error Average
    E = E/NUM_OF_TRAINING_DATA/N

    # back output -> intermediate
    
    for i in range(M):
      for j in range(N):
        nodes[1][i].addBackDelta(nodes[2][j].getOutDelta(nodes[0][j].getY()))

    # back intermediate -> input
    for i in range(N):
      for j in range(M):
        nodes[0][i].addBackDelta(nodes[1][j].getHiddenDelta())

    # update weights
    for i in range(N):
      for j in range(M):
        weights[0][i][j] = weights[0][i][j] - LEARNING_RATE * nodes[0][i].getY() * nodes[1][j].getHiddenDelta()

    for i in range(M):
      for j in range(N):
        weights[1][i][j] = weights[1][i][j] - LEARNING_RATE * nodes[1][i].getY() * nodes[2][j].getOutDelta(nodes[0][j].getY())

    if (cycle % 10 == 0):
      print('Cycle:\t{}, Error:\t{}'.format(cycle, E))

  # testing

