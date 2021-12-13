# DSA-RS
Document-level Sentiment Analysis Improved by Recommendation System

Public benchmarks: IMDb, Yelp2013, and Yelp2014

The model consists of three parts: 

an Autoencoder composed of an Attention-based Transformer, which will get a vector representation containing global potential information

a classifier based on the hierarchical BERT layers

the recommendation system, which introduces the obtained vector into the hierarchical BERT of the classifier so as to guide the prediction of the classifier and make the final result more accurate.
