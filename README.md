# high-component-ce-tools
To set up: 

`git clone https://github.com/juliayang/high-component-ce-tools.git`

`cd high-component-ce-tools/`

`python setup.py install`

The last step installs scikit-optimize for the Bayesian Charge Assigner and scikit-learn for the LassoCV and ElasticNet used in fitting the SparseGroupLasso. Additional dependencies include `pymatgen` for structure processing.

# Bayesian Charge Assigner 
The notebook, `examples/bayesian_optimization-charge-assignments.ipynb`, shows how to use Bayesian Optimization via Gaussian Processes to assign Mn2+, Mn3+, and Mn4+ charge states onto DFT-SCAN structures using d-orbital moments. All raw DFT-SCAN data are in `data/converged_structures.json`. 

# Sparse Group Lasso
The notebook, `examples/train-sparse-group-lasso.ipynb`, uses the feature matrix `data/fm.npy` and `data/e.npy` to do the regressing. SparseGroupLasso is used to enforce sparsity at the group-level (setting ECI to entire groups to 0), following the soft thresholding operator condition described by Simon, Friedman, Hastie, and Tibshirani 2011.

If you use this pacakge, please cite: 

J. H. Yang, T. Chen, L. Barroso-Luque, Z. Jadidi, G. Ceder, submitted (2022). 

Please email juliayang [at] berkeley [dot] edu with any questions. 
