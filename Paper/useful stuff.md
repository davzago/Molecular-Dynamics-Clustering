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