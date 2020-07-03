import subprocess
import os
import numpy as np
import re

def get_dense_array(path_to_pdb): # now cycle trough the files with os
  directory = os.fsencode(path_to_pdb)
  files = sorted(os.listdir(directory), key=lambda s : int("".join(re.findall(r'\d+', s.decode('utf-8'))))) # kind of ugly but useful to keep the files in the right order
  n = len(files)
  c = 0 # pos of the distance in the distance array
  distance_array = np.zeros(int(n*(n-1)/2)) # using gauss approach i calculate the number of unique pair distances
  resultFile = open("output_TMscore.txt","w")
  for i in range(0, n-1):
    for j in range(i+1, n):
      structure1 = files[i].decode('utf-8')
      structure2 = files[j].decode('utf-8')
      out = subprocess.check_output(['../Script/TMscore',
                                   path_to_pdb + '/' + structure1,
                                   path_to_pdb + '/' + structure2])
      distance_array[c] = parseRMSD(out)
      resultFile.write(str(i) + "-" + str(j) + ": " + str(distance_array[c]) + "\n")
      c += 1
  resultFile.close()
  return distance_array

def parseRMSD(output):
  return float(output.splitlines()[14].split()[5])