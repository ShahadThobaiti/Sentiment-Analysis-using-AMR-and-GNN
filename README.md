# Sentiment Analysis Using AMR and GNN

##### GNN Model

`GNN.ipynb`: The major compent of our model resides in this file. The AMR graph for each pre-processed tweet is generated priorly and stored as strings under `\data` directory. To feed AMR graphs into our GNN, we first build two bag-of-words (i.e., for nodes and edges) models based on our AMR graphs. Then for each AMR graph, we embedded its nodes, edges and edge_features as one-hot vectors. Before inputing these representations into our two-layer GNN offered by PyTorch Geometric library, we casted them into PyTorch's `Data` object.

`g_train.npy` and `g_test.npy ` are used to store large intermediate results (i.e., the edges and nodes features extracted from AMR graphs) that are necessary for our GNN training and testing.



##### Baseline Models

`Baselines.ipynb`: We compare our model with six baseline models known for sentiment analysis. This file contains five of them, including Decision Tree Classifier, Random Forest Classifer, Logistic Regressor, Gradient Boost Regressor and XGBoost Classifier. We also implemented lemmatization from NLTK to preprocess the data before training. We provide the same dataset for all models with 80:20 training-to-testing ratio.
