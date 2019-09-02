# EnsemblePursuit-- a sparse matrix factorization algorithm for extracting co-activating neurons from large-scale recordings

Ensemble Pursuit is a matrix factorization algorithm that places that extracts sparse neural components of co-activating cells. 

![alt text](https://github.com/mariakesa/EnsemblePursuit/blob/master/Figures/fig11.png){:height="100px" width="100px"}

The matrix U is a sparse matrix (because of an L0 penalty in the cost function) that encodes which neurons belong to a component. V is an average timecourse of these neurons, e.g. component time course.

For more details see the wiki https://github.com/mariakesa/EnsemblePursuit/wiki 

Ensembles learned using EnsemblePursuit from recordings in V1 have Gabor receptive fields. 

<img src="https://github.com/mariakesa/EnsemblePursuit/blob/master/Figures/ep_rec_fields.png" height="200" width="200">
![alt text](https://github.com/mariakesa/EnsemblePursuit/blob/master/Figures/ep_rec_fields.png)

Some ensembles are well explained by behavior PC's extracted from mouse orofacial movies.

![](https://github.com/mariakesa/EnsemblePursuit/blob/master/Figures/mouse.gif)


![alt text](https://github.com/mariakesa/EnsemblePursuit/blob/master/Figures/Behavior.png)


