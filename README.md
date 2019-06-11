# ACM_recsys_ML 2019
Mostly for the term project for TiA: Machine Learning at SNU spring 2019.
Will implement Factorization machine and some ANN (possibly RNN) solutions in addition to out own take on the problem.

## FM
Will be implemented as a tensorflow graph, in order to use tf.train.AdagradOptimizer. 
This is because the task given is to use that optimizer with some specific loss functions, BPR and TOP1.

### to run
~ = root folder for project
data:

0 - make sure data unziped in ~/data/

1 - run ~/data_code/extract_all_properties.py

2 - run ~/data_code/transform_item_FM.py

3 - run ~/data_code/transform_session_FM.py

training:

4 - train with ~/FM_code/test_placeholder.py

## RNN
 -- not yet implemented --
 
## UV
do steps 0-3 of FM to generate data.

1 - ~/UC_code/get_user_vectors.py

2 - ~/UC_code/max_user_variance_PCA.py
