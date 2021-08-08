# Udacity-Sparkify-Project

       
In this project, I run a case study on a pseudo company called Sparkify. 
Our goals are to identify these users and offer them attractive packages not to lose them. 
This history of these user’s interactions is provided as the dataset. 
This data contains unique user ids and a list of songs each listens to and their interaction with various.
In this project, we apply machine learning algorithms to analyze user behaviour and detect if they are 
going to downgrade or cancel their service, in industry know as "churn". 

The goal is to using machine leaning algorithm identify these type users offer then more interesting packages to keep them
 as an active users.

Our approach contains following steps:
* load and cleanse the raw data
* apply explanatory visualization techniques to identify user behaviour patterns and their correlation to churn event.  
* generate relevant features from raw data which can be used to train classification algorithm
* study the results of the classifiers and improve the models prediction results if possible


There are two dataset provided for out analysis:
* subset of 128 mB
* full dataset of 12 GB

For our analysis we have used the smaller dataset to provide the proof of concept. The larger dataset can be used for deploying 
this solution in cloud base environment. nether of the dataset are uploaded to this repository due to file size limit in Github.


Results Summery
----- 
In this article, we attempt to answer to a pseudo music app to keep its users from canceling or downgrading their paid subscription plan.
* we explored a subset of user activity log data. Our visualization provided insight into the type of interaction each user 
had with the App.
* The attempt to accumulate users’ interaction over time domain and generate features which embed temporal information and 
frequency of particular user interactions.
* We employed machine learning techniques to predict the churn even for each user given a set of accumulated features.
 Our results show up to 100% prediction accuracy when models built on top of featured generated via PCA algorithm.

* The high performance of these models was achieved on the subset of full data which contains only user logs for a period of 3 months. I would assume if the current pipeline applied to the full dataset the performance will drop.

The detail blog of these analysis is available here ["his-will-save-your-music-app-from-lossing-paid-users"](https://ramin-zohouri.medium.com/this-will-save-your-music-app-from-lossing-paid-users-91a07ac81ec5).


Content
-------
* [Sparkify Ipython Notebook](https://github.com/raminzohouri/Udacity-Sparkify-Project/blob/main/Sparkify.ipynb)
* [Helper functions that are used for making analysis](https://github.com/raminzohouri/Udacity-Sparkify-Project/blob/main/sparkify_utils.py)
* [Visualization results](https://github.com/raminzohouri/Udacity-Sparkify-Project/tree/main/images) 

Dependency
--------

Add the following python packages are required. There are four groups of required packages:
* ETL and Dataframe related
```python
pip3 install --user pandas
pip3 install --user numpy
```
* Visualisations
```python 
pip3 install --user seaborn
pip3 install --user matplotlib 
```
* Machine Learning
```python 
pip3 install --user spyspark
```

Usage
--------

* The `ipython notebooks` is interactive and can be run on your browser ro local machine.
    
More Information
----------------

* [Issues](https://github.com/raminzohouri/Udacity-Sparkify-Project/issues)


Useful Links
------------

* [scikit-learn](https://scikit-learn.org/stable/) - helps to find alternative machine learning solution.
* [pandas](https://pandas.pydata.org/) - helps to find more useful data analysis commands.
* [pyspark](https://spark.apache.org/docs/latest/api/python/index.html) Spark Python API.


Contributing
------------


License
-------

content of this survey is Copyright © 2008-2020 Joe Ferris and thoughtbot. It is free
software, and may be redistributed under the terms specified in the
[LICENSE] file.



----------------
