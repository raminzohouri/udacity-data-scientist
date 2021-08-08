# Disaster Response Project

In this project, we apply machine learning algorithms to analyze disaster data from [Figure Eight](https://www.figure-eight.com/) 
to build a model for an API that classifies disaster messages.

The models built from the [data set](https://appen.com/datasets/combined-disaster-response-data/) containing real messages
that were sent during disaster events. 

The machine learning pipeline categorizes these events so that we can send the messages to an appropriate disaster relief agency.

The Project includes a web app where an emergency worker can input a new message and get classification results in several categories.

 
Content
-------
* [ETL and Machine Learning Pipeline](https://github.com/raminzohouri/disaster_response_classifier/tree/master/disaster_response_classifier)
   
    *   [ETL:](https://github.com/raminzohouri/disaster_response_classifier/tree/master/disaster_response_classifier/process_data.py) loads original dataset, processes and store extracted features in the `SQLite` database.
   
    *   [Machine Learning Pipeline:](https://github.com/raminzohouri/disaster_response_classifier/tree/master/disaster_response_classifier/train_classifier.py) load training dataset from `SQLite` database. 
    Extract features using `NLP` methods and trains and evaluate a multi-class classification algorithm. 
    Then saves trained model.
    
* [Data:](https://github.com/raminzohouri/disaster_response_classifier/tree/master/data) contain original dataset and generated database.

* [Models:](https://github.com/raminzohouri/disaster_response_classifier/tree/master/models) contains trained models.

* [Notebooks:](https://github.com/raminzohouri/disaster_response_classifier/tree/master/notebooks) contains `Ipython Notebooks` from hands on step
by step experimenting with ETL and classification pipeline.

* [Webapp:](https://github.com/raminzohouri/disaster_response_classifier/tree/master/webapp) contains a web application program
which loads and visualized extracted features stored in `SQLite` database. Plus loads a trained model 
and classifies a text message into multi-class and advises some organizations to connect to.

Dependency
--------

Add the following python packages are required. There are four groups of required packages:
* ETL and Dataframe related
```python
pip3 install --user pandas
pip3 install --user numpy
pip3 install --user sqlalchemy
pip3 install --user nltk  
```
* Visualisations
```python 
pip3 install --user seaborn
pip3 install --user matplotlib 
```
* Machine Learning
```python 
pip3 install --user sklean
pip3 install --user joblib 
```
* Webapp builder
```python 
pip3 install --user plotly
pip3 install --user flask 
```

Usage
--------

* The `ipython notebooks` are interactive and can be run on your browser ro local machine.
Fill free to try them out and add your changes and ideas.
* For running the project on you local machine the following steps have to be followed:
    *  clone the repository:
        * `git clone git@github.com:raminzohouri/disaster_response_classifier.git`
    *  To build the model and run the webapp from project root execute following command:
        * `python3 process python3 disaster_response_classifier/process_data.py`
            * The database will contain one table, `DisasterResponseData`.
        * `python3 process python3 disaster_response_classifier/train_classifier.py`
            * The model will be saved as compressed format `*.lzma`
        *  `python3 webapp/run.py`
            * The webapp runs with default values for model and database stored in `modesl` and `data` directories.
            * The webapp opens automatically in this url `http://127.0.0.1` with a random chosen port.
            * It is possible to supply alternative `db_file` and `model_file` as arguments.
            * You can submit question to webapp and receive results.
            * You can click on the valid results and directed to relevant organisations.  
    * all the input paths can be modified accordingly. user `--help` option to get list of arguments.
    
More Information
----------------

* [Issues](https://github.com/raminzohouri/stackoverflow_survey_analysis/issues)


Useful Links
------------

* [scikit-learn](https://scikit-learn.org/stable/) - helps to find alternative machine learning solution.
* [pandas](https://pandas.pydata.org/) - helps to find more useful data analysis commands.
* [nltk](https://www.nltk.org/) Natural Language Toolkit is a platform for building Python programs to work with human language data.
* [Flask](https://flask.palletsprojects.com/en/1.1.x/) Wep development platform using python.
* [Plotly/Dash](https://plotly.com/dash/) Is the most downloaded, trusted framework for building ML & data science web apps. 


Contributing
------------
Feel free to fork this repository and or make pull requerst to in order to add new features.

License
-------

content of this repository is Copyright © 2020-2020 Ramin Zohouri. It is free
software, and may be redistributed under the terms specified in the
[LICENSE] file.



----------------
