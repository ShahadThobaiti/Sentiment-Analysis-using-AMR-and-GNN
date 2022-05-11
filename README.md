# Sentiment Analysis Using AMR and GNN

Graph Neural Networks are a promising approach in Natural Language Processing that have applications in dependency parsing and question answering systems [1][2]. This motivates us to build a text classification model that utilizes Abstract Meaning Representation (AMR) and a Graph Neural Networks (GNN). We use the AMR to parse sentences into graph structure to be trained in GNN. For evaluation, we train our proposed model with a sentiment analysis task and compare it to baseline models such as Logistic Regression, as well as with state-of-the-art models such as BERT classifier. 

The poster of this work can be found [here](Poster.pdf).



##### Preprocessing and AMR Representation  of Tweets:
`Preprocessing_AMR.ipynb` include the code for preprocessing the tweet to grammatically correct it using OpenAI's GPT-3 via Davinci engine (the most capable engine). After that we produce the AMR graph using AMRlib library. 


##### GNN Model

`GNN.ipynb`: The major compent of our model resides in this file. The AMR graph for each pre-processed tweet is generated priorly and stored as strings under `\data` directory. To feed AMR graphs into our GNN, we first build two bag-of-words (i.e., for nodes and edges) models based on our AMR graphs. Then for each AMR graph, we embedded its nodes, edges and edge_features as one-hot vectors. Before inputing these representations into our two-layer GNN offered by PyTorch Geometric library, we casted them into PyTorch's `Data` object.

`g_train.npy` and `g_test.npy ` are used to store large intermediate results (i.e., the edges and nodes features extracted from AMR graphs) that are necessary for our GNN training and testing.


##### Baseline Models

`Baselines.ipynb`: We compare our model with six baseline models known for sentiment analysis. This file contains five of them, including Decision Tree Classifier, Random Forest Classifer, Logistic Regressor, Gradient Boost Regressor and XGBoost Classifier. We also implemented lemmatization from NLTK to preprocess the data before training. We provide the same dataset for all models with 80:20 training-to-testing ratio.


