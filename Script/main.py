import argparse

import parsing
import matrix

### $ python3 stru.py contact_maps_paths.txt -conf params.ini -out_dir results/ -tmp_dir /tmp

parser = argparse.ArgumentParser(description='Hierarchical clustering using RING data.')
parser.add_argument('data_path', help='File with the path to RING contact map files (edge files)')
parser.add_argument('-conf', help='Configuration file with algorithm parameters')
parser.add_argument('-out_dir', help='Output directory')
parser.add_argument('-tmp_dir', help='Temporary file directory')
args = parser.parse_args()
snapSet = parsing.parse(args.data_path)

'''
### DISTINCT EDGES AND OCCURENCE COUNT IN ALL SNAPSHOT
count = dict()
for i in range(0, len(snapSet)-1):
    for s in snapSet[i].edges:
        if((str(s[0]) + "-" + str(s[1])) in count):
            count[str(s[0]) + "-" + str(s[1])] += 1
        else:
            count[str(s[0]) + "-" + str(s[1])] = 1

for i in count:
    print(i, ": ", count[i])

print(len(count))
'''