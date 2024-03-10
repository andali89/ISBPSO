# ISBPSO Implementation

This is an implementation of the improved sticky binary PSO (ISBPSO) algorithm 
for feature selection (FS) for high-dimensional data. ISBPSO uses a new initialization strategy 
using the feature weighting information based on mutual information, and uses a dynamic
bits masking strategy for gradually reducing the search space during the evolutionary process.
For a detailed description of the method please refer to 


> Li, A.-D., Xue, B., & Zhang, M. (2021). Improved binary particle swarm optimization for feature selection with new initialization and search space reduction strategies. Applied Soft Computing, 106, 107302.  [https://doi.org/10.1016/j.asoc.2021.107302](https://doi.org/10.1016/j.asoc.2021.107302) [[BibTeX](https://andali89.github.io/homepage/bibfiles/Li2021IBPSOFS.bib)] [[PDF](https://andali89.github.io/homepage/pubs/2021_ISBPSO.pdf)]

The source code is in the [src](./src/) folder. An illustration to run the algorithm is implemented in [./ISBPSO/src/fs/RunPSO.java](./ISBPSO/src/fs/RunPSO.java). Please note that the code requires the jar file ([./ISBPSO/lib/weka.jar](./ISBPSO/lib/weka.jar)) of [Weka](https://www.cs.waikato.ac.nz/ml/weka/), Machine Learning Software in Java. Please make sure it is added in the libiray.  read me
