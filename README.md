# ML_FakeVsRealNews
A project that use TFIDF to train and create a model that categorizes fake and real news from the presidential election in 2016 database.

This project use Natural Language tool kit (nltk) to preprocess the news text and Scikit-learn (Sklearn) library to vectorize each news into TFIDF vectors and apply different classifying methods.

There are 3 classification methods in this project: Logistic regression, Naive Bayer, and SVM. Naive Bayer gives the least accuracy (84%), following Logistic regression with 90%. The most accuracy resulted by SVM with 94%. However, run time of SVM is the longest with more than 12 hours while Logistic regression and Naive Bayer only take a couple minutes.
