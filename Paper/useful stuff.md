Euclidean distance is usually not good for sparse data 

I'd suggest starting with [Cosine distance](http://en.wikipedia.org/wiki/Cosine_similarity), not Euclidean, for any data with most vectors nearly orthogonal, x⋅y≈x⋅y≈ 0.
To see why, look at |x−y|2=|x|2+|y|2−2 x⋅y|x−y|2=|x|2+|y|2−2 x⋅y.
If x⋅y≈x⋅y≈ 0, this reduces to |x|2+|y|2|x|2+|y|2: a crummy measure of distance, as Anony-Mousse points out.

Cosine distance amounts to using x/|x|x/|x|, or projecting the data onto the surface of the unit sphere, so all |x||x| = 1. Then |x−y|2=2−2 x⋅y|x−y|2=2−2 x⋅y
a quite different and usually better metric than plain Euclidean. x⋅yx⋅y may be small, but it's not masked by noisy |x|2+|y|2|x|2+|y|2.

x⋅yx⋅y is mostly near 0 for sparse data. For example, if xx and yy each have 100 terms non-zero and 900 zeros, they'll both be non-zero in only about 10 terms (if the non-zero terms scatter randomly).

Normalizing xx /= |x||x| may be slow for sparse data; it's fast in [scikit-learn](http://scikit-learn.org/stable/developers/utilities.html#efficient-routines-for-sparse-matrices).

Summary: start with cosine distance, but don't expect wonders on any old data.
Successful metrics require evaluation, tuning, domain knowledge.

If we use PCA the data gets normalized (with sklearn implementation just centered) by doing this we lose the sparsity of the matrix, the truncatdSVD method is efficient on sparse data hence it is computationally better 

##### Yule(X)(frataxin):

mutual info score between RMSD clustering and contact clustering: 0.5813110726407498
RandIndex between RMSD clustering and contact clustering: 0.28832878188893984

##### Hamming(X)(frataxin):

maximum RMSD silhouette: 0.3109240910863861 with 4 clusters
maximum contacts silhouette: 0.04939135337801037 with 9 clusters
mutual info score between RMSD clustering and contact clustering: 0.5468241872908568
RandIndex between RMSD clustering and contact clustering: 0.26418832867433695

##### Pearson(SVD(vector_edges))(antibody)

maximum RMSD silhouette: 0.3998577252872483 with 4 clusters
maximum contacts silhouette: 0.04784837788034364 with 50 clusters
RandIndex between RMSD clustering and contact clustering: 0.01834481813657844

##### Pearson(SVD(vector_edges))(vcb)

maximum RMSD silhouette: 0.35786785137279037 with 8 clusters
maximum contacts silhouette: 0.09848855327015538 with 4 clusters
RandIndex between RMSD clustering and contact clustering: 0.38526946876946355

##### Pearson(SVD(vector_edges))(frataxin)

maximum RMSD silhouette: 0.3109240910863861 with 4 clusters
maximum contacts silhouette: 0.13009748877357985 with 9 clusters
RandIndex between RMSD clustering and contact clustering: 0.05760775683294067

##### Yule(vector_edges)(antibody)

maximum contacts silhouette: 0.06060011114161691 with 4 clusters
RandIndex between RMSD clustering and contact clustering: -0.05408512232523832

##### Yule(vector_edges)(vcb)

maximum contacts silhouette: 0.1448083464926949 with 4 clusters
RandIndex between RMSD clustering and contact clustering: 0.3748386167742934

##### Yule(vector_edges)(frataxin)

maximum contacts silhouette: 0.15613449042331937 with 4 clusters
RandIndex between RMSD clustering and contact clustering: 0.10913881222964728



