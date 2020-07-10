import argparse
import subprocess
import os
import numpy as np
import re
import RMSD

def get_distance_matrix(path_to_pdb): # now cycle trough the files with os
  directory = os.fsencode(path_to_pdb)
  files = sorted(os.listdir(directory), key=lambda s : int("".join(re.findall(r'\d+', s.decode('utf-8'))))) # kind of ugly but useful to keep the files in the right order
  n = len(files)
  name = path_to_pdb.split('/')[6]
  print(name)
  distance_array = np.zeros((n,n)) # using gauss approach i calculate the number of unique pair distances
  resultFile = open(name + "_RMSD.txt","w")
  for i in range(0, n):
    for j in range(0, n):
      structure1 = files[i].decode('utf-8')
      structure2 = files[j].decode('utf-8')
      out = subprocess.check_output(['../Script/TMscore',
                                   path_to_pdb + '/' + structure1,
                                   path_to_pdb + '/' + structure2])
      distance_array[i,j] = RMSD.parseRMSD(out)
      resultFile.write(str(i) + "-" + str(j) + ": " + str(distance_array[i,j]) + "\n")
  resultFile.close()
  return distance_array

parser = argparse.ArgumentParser(description='Hierarchical clustering using RING data.')
parser.add_argument('data_path', help='path to the MD folder')
path = parser.parse_args().data_path

directory = os.fsencode(path)
dirz = os.listdir(directory)
for i in range(0,len(dirz)):
    dirz[i] = path + '/' + dirz[i].decode('utf-8') + '/pdb'

for p in dirz:
    get_distance_matrix(p)


