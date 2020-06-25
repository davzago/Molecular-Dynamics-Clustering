import subprocess

def get_dense_array(): # now cycle trough the files with os
  out = subprocess.check_output(['/home/davide/Documenti/structural_bio/Script/TMscore',
   '../MD_simulations/antibody/pdb/6J6Y_1_ms_1K0.1.pdb',
    '../MD_simulations/antibody/pdb/6J6Y_1_ms_1K0.2.pdb'])
  parseRMSD(out)

def parseRMSD(output):
  return float(output.splitlines()[14].split()[5])