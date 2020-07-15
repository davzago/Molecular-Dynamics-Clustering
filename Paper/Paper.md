# Hierarchical clustering of protein's contact maps

To run the code the following libraries are required:

- Argparse
- Numpy
- Scikit learn
- Scipy
- Yellowbrick

This python script returns the hierarchical clustering of the contact maps given in input by giving in output:

- A representative contact map for each cluster created
- A dendogram showing the clusters and where the tree is cut
- A file txt containing the label of each snapshot
- A file txt containing the distance matrix used in order to obtain the clustering
- A file containing the most important contacts in each cluster

**The code must be ran from the Script folder** and to obtain the outputs the command is:

``` shell
python3 main.py contact_maps_paths.txt
```

Where *contact_maps_paths.txt* is the file containing the path to the edge files (snapshots to clusterize).

## Evaluation of the clustering

To evaluate the obtained clustering we compared it to the hierarchical clustering obtained using the pairwise RMSD which is computed using *TM-Score.cpp*.

In order to compute the pairwise RMSD for a new MD the command is:

``` shell
python3 main.py contact_maps_paths.txt -RMSD_path path_to_RMSD_file -path_to_pdb path_to_pdb_folder 
```

it is also possible to change the path of the output directory using *-out_dir*.

To use the *-path_to_pdb* argument is necessary to compile *TMscore.cpp* using the linux command:

``` shell
g++ -static -O3 -ffast-math -lm -o TMscore TMscore.cpp
```

The '-static' flag should be removed on Mac OS, which does not support building static executables.

When using *-path_to_pdb* the program will write a *RMSD.txt* file in *out_dir/contact_maps_paths* which is the file to pass as argument when using *-RMSD_path*.

*-RMSD_path* makes so the program computes the clustering based on the RMSD and compares it with the contact map clustering using *adjusted_rand_score* from scikit learn

### Adjusted RandIndex

The Rand Index computes a similarity measure between two clusterings by considering all pairs of samples and counting pairs that are assigned in the same or different clusters in the predicted and true clusterings, the raw RI score is then “adjusted for chance” into the ARI score using the following scheme:
$$
ARI = \frac{(RI - Expected_{RI})}{ (max(RI) - Expected_{RI})}
$$
Note that this measure is severe since is usually used to check the obtained clustering against the ground truth.

## Clustering Algorithm

### PCA

### Elbow and Silhouette



