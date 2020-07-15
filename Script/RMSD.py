import subprocess
import os
import numpy as np
import re
import math

def parseRMSD(output):
  return float(output.splitlines()[14].split()[5])

def get_distance_matrix_from_file(path_to_file):
  matrix = []
  with open(path_to_file) as f:
    lines = [line.rstrip() for line in f]
    size = int(math.sqrt(len(lines)))
    matrix = np.zeros((size,size))
    for line in lines:
      split1 = line.split(':')
      i = int(split1[0].split('-')[0])
      j = int(split1[0].split('-')[1])
      dist = float(split1[1])
      matrix[i][j] = dist
  return matrix