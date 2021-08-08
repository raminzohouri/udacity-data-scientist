#!/usr/bin/env python

# import libraries
import re
from matplotlib import pyplot as plt
import seaborn as sns

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as pysTypes
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler, MinMaxScaler, PCA
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics

sns.set_style("ticks")
sns.set_color_codes()
sns.set(font_scale=1)


def cleanse_numericals_and_categoricals(df):
    """
    :param df: Spark Dataframe
    :return df: Spark Dataframe
    """
    print("\n number missing values for each column")
    df.select(
        [F.count(F.when(F.isnan(c) | F.col(c).isNull() | (F.length(c) == 0), c)).alias(c) for c in df.columns]).show()
    print("\n size before clean up:{}".format(df.count()))
    df = df.dropDuplicates()
    df = df.filter(F.col("userId") != "")

    print("\n Add user device")
    user_devices = {"Windows": "Windows", "iPad": "iPad", "iPhone": "iPhone", "Macintosh": "Machintosh",
                    "compatible": "Windows", "X11": "Linux"}
    regs = F.udf(lambda x: user_devices[re.findall(r"\(\w*", x)[0][1:]])
    df = df.withColumn("userDevice", regs(F.col("userAgent")))

    print("\n size after clean up:{}".format(df.count()))
    print(
        "\n notice removing missing values from  [userId] also cleans up other columns such as [firstName, gender, lastName, location, registration, userAgent] ")
    df.select(
        [F.count(F.when(F.isnan(c) | F.col(c).isNull() | (F.length(c) == 0), c)).alias(c) for c in df.columns]).show()

    print("\n cast userId type to long")
    df = df.withColumn("userId", F.col("userId").cast(pysTypes.LongType()))

    print("\n generate month, monthDay, hour, weekDay from ts")
    df = df.withColumn("month", F.month(F.from_unixtime(F.col("ts") * 0.001).cast(pysTypes.DateType())))
    df = df.withColumn("monthDay", F.dayofmonth(F.from_unixtime(F.col("ts") * 0.001).cast(pysTypes.DateType())))
    df = df.withColumn("hour", F.hour(F.from_unixtime(F.col("ts") * 0.001)))
    df = df.withColumn("weekDay", F.dayofweek(F.from_unixtime(F.col("ts") * 0.001)))
    df = df.withColumn("date", F.from_unixtime(F.col("ts") * 0.001).cast(pysTypes.DateType()))

    print("\n generate from location value with only the name of the state")
    df = df.withColumn("location", F.split(F.col("location"), ",").getItem(1))

    print("\n there is no NextPage event with length ==0")
    df.filter("page == 'NextSong'").filter(
        F.isnan("length") | F.col("length").isNull() | (F.length("length") == 0)).show()

    print("\n fill all nan values for [artist, length, song] with zero")
    df = df.fillna({"artist": "unknown", "length": 0, "song": "unknown"})

    return df


# my helper functions
def generate_churn_label(df):
    """
    Calculates Churn from existing page events
    follwing event are considered as churn ["Submit Downgrade", "Cancellation Confirmation"]
    Adds churn column
    :param df: spark DataFrame
    :return df: spark DataFrame
    """
    if "churn" in df.columns:
        df = df.drop("churn")
    churn_event = F.udf(lambda x: int(x in ["Cancellation Confirmation"]), pysTypes.IntegerType())
    df_churn = df.select(["userId", "page"]).withColumn("churn", churn_event("page")).groupby("userId").agg(
        F.sum("churn").alias("churn")).withColumn("churn", F.when(F.col("churn") > 0, 1).otherwise(0))
    df = df.join(df_churn, on="userId")
    return df


def plot_churn_frequency(df, plot_columns):
    """
     given a dataframe and column name, plot visualize column values color coded via churn event
    :param df:
    :param plot_columns:
    :return:
    """
    for c, s in plot_columns.items():
        dfp = df.groupBy(c, "churn").count().toPandas()
        fz = s
        plt.figure(figsize=fz)
        ax = sns.barplot(x=c, y="count", hue="churn", data=dfp);
        h, l = ax.get_legend_handles_labels()
        ax.legend(h, ["Active", "Canceled"], title="user status")
        ax.set_ylabel("number of users")
        ax.set_xticklabels(labels=ax.get_xticklabels(), size=15, rotation=90)
        plt.savefig("".join([c, "-barplot.png"]))


def plots_and_prints(df_labelled):
    """

    :param df_labelled:
    :return:
    """
    print(" Number of users: {}, number of sessions: {}".format(df_labelled.select("userId").dropDuplicates().count(),
                                                                df_labelled.select(
                                                                    ["userId", "sessionId"]).dropDuplicates().count()))
    print("\n top 20 users with maximum number of sessions")
    df_labelled.groupBy(["userId", "sessionId"]).count().sort(F.desc("count")).show(20)

    print("distribution of the Cancellation based on  weekDay, monthDay, hour of the day")
    dfp = df_labelled.filter("page=='Cancellation Confirmation'").select(["weekDay", "monthDay", "hour"]).toPandas()
    fig = plt.figure()
    ax = plt.gca()
    ax.set_title("distribution of churn event over hours of each day")
    dfp["hour"].value_counts().sort_index().plot.bar(figsize=(8, 5), color="g", ax=ax, alpha=0.75, rot=0);
    fig = plt.figure()
    ax = plt.gca()
    ax.set_title("distribution of churn event over day of each week")
    dfp["weekDay"].value_counts().sort_index().plot.bar(figsize=(8, 5), color="g", ax=ax, alpha=0.75, rot=0);
    fig = plt.figure()
    ax = plt.gca()
    ax.set_title("distribution of churn event over day of each month")
    dfp["monthDay"].value_counts().sort_index().plot.bar(figsize=(8, 5), color="g", ax=ax, alpha=0.75, rot=0);

    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()
    ax.set_title("accumulated-sessions-length")
    h, l = ax.get_legend_handles_labels()
    ax.legend(h, ["Active", "Canceled"], title="user status")
    df_labelled.filter("churn == 0").select(["userId", "length", "sessionId"]).groupBy("userId", "sessionId").agg(
        F.sum("length").alias("Active")).toPandas()["Active"].plot(kind="hist", legend="Active", ax=ax, rwidth=0.7);
    df_labelled.filter("churn == 1").select(["userId", "length", "sessionId"]).groupBy("userId", "sessionId").agg(
        F.sum("length").alias("Canceled")).toPandas()["Canceled"].plot(kind="hist", color="g", legend="Canceled", ax=ax,
                                                                       rwidth=0.7, );

    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()
    ax.set_title("number of essions per user")
    h, l = ax.get_legend_handles_labels()
    ax.legend(h, ["Active", "Canceled"], title="user status")
    df_labelled.filter("churn ==0").select(["userId", "sessionId"]).drop_duplicates().groupBy(
        "userId", ).count().withColumn("Active", F.col("count")).toPandas()["Active"].plot(kind="hist", legend="Active",
                                                                                           color="b", rwidth=0.7, );
    df_labelled.filter("churn ==1").select(["userId", "sessionId"]).drop_duplicates().groupBy(
        "userId", ).count().withColumn("Canceled", F.col("count")).toPandas()["Canceled"].plot(kind="hist",
                                                                                               legend="Canceled",
                                                                                               color="g", rwidth=0.7);

    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()
    ax.set_title("max #itemsInsessions per user")
    h, l = ax.get_legend_handles_labels()
    ax.legend(h, ["Active", "Canceled"], title="user status")
    df_labelled.filter("churn ==0").groupBy(["userId", "sessionId"]).agg(
        F.max("itemInSession").alias("Active")).toPandas()["Active"].plot(kind="hist", color="b", legend="Active",
                                                                          rwidth=0.7, );
    df_labelled.filter("churn ==1").groupBy(["userId", "sessionId"]).agg(
        F.max("itemInSession").alias("Canceled")).toPandas()["Canceled"].plot(kind="hist", color="g", legend="Canceled",
                                                                              rwidth=0.7, );

    print("\n plot churn frequency for categorical features")
    plot_churn_frequency(df_labelled,
                         {"auth": (8, 5), "gender": (8, 5), "level": (8, 5), "page": (28, 10), "userDevice": (8, 5),
                          "location": (32, 10)})

    dfp = df_labelled.dropDuplicates(["userId"]).groupby(["churn"]).count().toPandas().set_index("churn")
    ax = sns.barplot(x=dfp.index, y="count", data=dfp);
    ax.set_xticklabels(["Active", "Canceled"]);

    print("\n plot churn frequency for time features")
    plot_churn_frequency(df_labelled, {"month": (8, 5), "monthDay": (15, 5), "hour": (15, 5), "weekDay": (8, 5)})


def generate_page_based_features(df):
    """
    the following features are generated base userId and page activities 
        * rollAdvertPageMonth: double (nullable = false)
        * settingsPageMonth: double (nullable = false)
        * downgradePageMonth: double (nullable = false)
        * nextSongPageMonth: double (nullable = false)
        * errorPageMonth: double (nullable = false)
        * aboutPageMonth: double (nullable = false)
        * upgradePageMonth: double (nullable = false)
        * homePageMonth: double (nullable = false)
        * logoutPageMonth: double (nullable = false)
        * addtoPlaylistPageMonth: double (nullable = false)
        * thumbsDownPageMonth: double (nullable = false)
        * thumbsUpPageMonth: double (nullable = false)
        * saveSettingsPageMonth: double (nullable = false)
        * addFriendPageMonth: double (nullable = false)
        * submitUpgradePageMonth: double (nullable = false)
        * helpPageMonth: double (nullable = false)
        * submitDowngradePageMonth: double (nullable = false)
       
    :param df: Spark DataFrame
    :return df: Spark DataFrame
    """
    print("\n add features based on page activities")
    tmdf = df.select(["userId", "month", "page"]).filter(
        ~F.col("page").isin(["Cancel", "Cancellation Confirmation"])).groupBy("userId", "month", ).pivot(
        "page").count().fillna(0)
    for c in tmdf.columns:
        if c not in ["userId", "month", "page"]:
            tmdf = tmdf.withColumnRenamed(c, "".join([c.replace(" ", ""), "PageMonth"]))
    tmdf = tmdf.drop("month", "page")
    df = tmdf.join(df, on="userId")
    return df


def generate_dummy(df, dum_col):
    """
    get dummy variables for given column
    :param df: Spark DataFrame
    :return df: Spark DataFrame
    """
    print("\n add dummy features based {}".format(dum_col))
    tmdf = df.groupBy("userId").pivot(dum_col).count().fillna(0)
    for c in tmdf.columns:
        if c not in ["userId", dum_col]:
            tmdf = tmdf.withColumnRenamed(c, "".join([dum_col, c.replace(" ", "")]))
    df = tmdf.join(df, on="userId")
    df = df.drop(dum_col)
    return df


def generate_dummies_based_features(df):
    """
    generate dummy features for each user from following columns 
        * auth: string (nullable = true)
        * gender: string (nullable = true)
        * level: string (nullable = true)        
        * userDevice: string (nullable = true)
       
    :param df: Spark DataFrame
    :return df: Spark DataFrame
    """

    df = generate_dummy(df, "gender")
    df = generate_dummy(df, "level")
    df = generate_dummy(df, "userDevice")

    return df


def generate_session_based_features(df):
    """
    the following features are generated base userId and sessionId 
        * avgSessionsMonth: double (nullable = true)
        * avgSessionMonthDuration: double (nullable = true)
        * avgSessionitemsMonth: double (nullable = true)
        * avgSessionsDay: double (nullable = true)
        * avgSessionDayDuration: double (nullable = true)
        * avgSessionitemsDay: double (nullable = true)
        * activeDuration: double(nullable = true)
        * avgLength: double(nullable = true)
    :param df: Spark DataFrame
    :return df: Spark DataFrame
    """

    # monthly features
    print("\n add average number of session in each month per user 'avgSessionsMonth'")
    tmdf1 = df.groupby("userId", "month").agg(F.countDistinct("sessionId").alias("avgSessionsMonth")).groupBy(
        "userId").avg("avgSessionsMonth").withColumnRenamed("avg(avgSessionsMonth)", "avgSessionsMonth")

    print("\n add average session duration in each month per user 'avgSessionMonthDuration' ")
    tmdf2 = df.groupby("userId", "month", "sessionId").agg(F.max("ts").alias("session_end"),
                                                           F.min("ts").alias("session_start")).withColumn(
        "avgSessionMonthDuration", (F.col("session_end") - F.col("session_start")) * 0.001).groupby("userId",
                                                                                                    "month").avg(
        "avgSessionMonthDuration").groupby("userId").agg(
        F.mean("avg(avgSessionMonthDuration)").alias("avgSessionMonthDuration"))

    print("\n add average number of session items in each month per user 'avgSessionitemsMonth'")
    tmdf3 = df.groupby("userId", "sessionId", "month").agg(
        F.max("itemInSession").alias("avgSessionitemsMonth")).groupBy("userId").avg(
        "avgSessionitemsMonth").withColumnRenamed("avg(avgSessionitemsMonth)", "avgSessionitemsMonth")

    # daily features
    print("\n add average number of session in each day per user 'avgSessionsDay'")
    tmdf4 = df.groupby("userId", "date").agg(F.countDistinct("sessionId").alias("avgSessionsDay")).groupBy(
        "userId").avg("avgSessionsDay").withColumnRenamed("avg(avgSessionsDay)", "avgSessionsDay")

    print("\n add average number of session duration in each day 'avgSessionDayDuration'")
    tmdf5 = df.groupby("userId", "date", "sessionId").agg(F.max("ts").alias("session_end"),
                                                          F.min("ts").alias("session_start")).withColumn(
        "avgSessionDayDuration", (F.col("session_end") - F.col("session_start")) * 0.001).groupby("userId", "date").avg(
        "avgSessionDayDuration").groupby("userId").agg(
        F.mean("avg(avgSessionDayDuration)").alias("avgSessionDayDuration"))

    print("\n add average session items in each day per user 'avgSessionitemsDay'")
    tmdf6 = df.groupby("userId", "sessionId", "date").agg(F.max("itemInSession").alias("avgSessionitemsDay")).groupBy(
        "userId").avg("avgSessionitemsDay").withColumnRenamed("avg(avgSessionitemsDay)", "avgSessionitemsDay")

    print("\n add user's active time duration since registration 'activeDuration'")
    tmdf7 = df.groupBy("userId").agg((F.max(F.col("ts")) - F.min(F.col("registration"))) * 0.001).withColumnRenamed(
        "((max(ts) - min(registration)) * 0.001)", "activeDuration")

    print("\n add average length song which a users have listened ")
    tmdf8 = df.groupBy("userId").agg(F.avg("length").alias("avgLength"))

    return tmdf1.join(tmdf2, on="userId").join(tmdf3, on="userId").join(tmdf4, on="userId").join(tmdf5,
                                                                                                 on="userId").join(
        tmdf6, on="userId").join(tmdf7, on="userId").join(tmdf8, on="userId").join(df, on="userId")


def plot_features(df):
    """

    :param df:
    :return:
    """
    dfp = df.toPandas()

    column_list = ["avgSessionsMonth", "avgSessionMonthDuration", "avgSessionitemsMonth", "avgSessionsDay",
                   "avgSessionDayDuration", "avgSessionitemsDay", "activeDuration",
                   "AboutPageMonth", "AddFriendPageMonth", "AddtoPlaylistPageMonth", "DowngradePageMonth",
                   "ErrorPageMonth", "HelpPageMonth", "HomePageMonth", "LogoutPageMonth",
                   "NextSongPageMonth", "RollAdvertPageMonth", "SaveSettingsPageMonth", "SettingsPageMonth",
                   "SubmitDowngradePageMonth", "SubmitUpgradePageMonth", "ThumbsDownPageMonth",
                   "ThumbsUpPageMonth", "UpgradePageMonth", "itemInSession", "avgLength"]
    for c in column_list:
        fig = plt.figure(figsize=(8, 5))
        ax = fig.gca()
        h = dfp[dfp.churn == 0.0][c].plot.hist(color='b', ax=ax, )
        h = dfp[dfp.churn == 1.0][c].plot.hist(color='g', ax=ax)
        h, l = ax.get_legend_handles_labels()
        ax.set_xlabel(c)
        ax.legend(h, ["Active", "Canceled"], title="user status")

    fig = plt.figure(figsize=(30, 25))
    ax = fig.gca()
    h = dfp.hist(ax=ax)


def feature_scaling(df):
    """
    apply MinMax scaling on the desired features
    :param df:
    :return:
    """
    feature_cols = df.drop("userId", "churn").columns
    df = df.withColumn("label", df["churn"].cast(pysTypes.DoubleType()))
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="feature_vec")

    # pyspark default name for features vector column: 'featuresCol'
    minmaxscaler = MinMaxScaler(inputCol="feature_vec", outputCol="features")

    df = assembler.transform(df)
    minmaxscaler_model = minmaxscaler.fit(df)
    scaled_df = minmaxscaler_model.transform(df)
    return scaled_df


def custom_evaluation(clf_pred_results, model_name):
    """
    Perform custom evaluation of predictions
       - inspect with PySpark.ML evaluator (will use for pipeline)
       - use RDD-API; PySpark.MLLib to get metrics based on predictions 
       - display confusion matrix
    :param clf_pred_results: Spark Dataframe
    :param model_name: String name of classifier
    """

    pr = BinaryClassificationEvaluator(metricName="areaUnderPR")

    pr_auc = pr.evaluate(clf_pred_results)
    print("\n BinaryClassificationEvaluator results:")
    print(f"{model_name} -> PR AUC: {pr_auc}")

    predictionRDD = clf_pred_results.select(["label", "prediction"]).rdd.map(lambda line: (line[1], line[0]))
    metrics = MulticlassMetrics(predictionRDD)
    print("\n MulticlassMetrics results:")
    print(f"{model_name}\n | precision = {metrics.precision()}")
    print(f" | recall = {metrics.recall()}\n | F1-Score = {metrics.fMeasure()}")

    conf_matrix = metrics.confusionMatrix().toArray()
    sns.set(font_scale=1.4)  # for label size
    ax = sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 16})
    ax.set(xlabel="Predicted Label", ylabel="True Label", title="Confusion Mtx")
    plt.show()


def train_predict_models(train_df, test_df, model_name, featuresCol='features', labelCol='label', ):
    """
    build the classification model and train the and test on the given dataframe and visualize the results
    :params train_df: Spark Dataframe
    :params test_df: Spark Dataframe
    :param model_name: classification model name
    :param featureCol: String name of the feature column used via classifier
    :param labelCol: String name of the label column used vi classifier
    """

    if model_name == "LogisticRegression":
        ml_model = LogisticRegression(featuresCol=featuresCol, labelCol=labelCol)
    elif model_name == "RandomForestClassifier":
        ml_model = RandomForestClassifier(featuresCol=featuresCol, labelCol=labelCol)
    elif model_name == "GBTClassifier":
        ml_model = GBTClassifier(featuresCol=featuresCol, labelCol=labelCol)
    else:
        print("\n expected classifiers are: [GBTClassifier, RandomForestClassifier, LogisticRegression]")

    clf_model = ml_model.fit(train_df)
    results_clf = clf_model.transform(test_df)
    custom_evaluation(results_clf, model_name)


def plot_correlation(df, feature_column, plt_title="correlation of scaled features", set_tick_label=True):
    """
    plot correlation of scaled features 
    :param df: Spark Dataframe
    :param feature_column: String name of the feature column
    :param plt_title: String plot title
    :param set_tick_label: bool if True set labels for x and x ticks
    """
    pearsonCorr = Correlation.corr(df, feature_column, "pearson").collect()[0][0]
    fig = plt.figure(figsize=(25, 25))
    ax = fig.gca()
    ax.set_title(plt_title)
    sns.set(font_scale=3)
    p = sns.heatmap(pearsonCorr.toArray(), ax=ax, robust=True, cmap="YlGnBu")
    if set_tick_label:
        df_cols = scaled_df.drop("userId", "features", "churn", "label", "feature_vec").columns
        p.set_yticklabels(df_cols, size=15, rotation=0)
        p.set_xticklabels(df_cols, size=15, rotation=90)


def find_best_pca_param(scaled_df, init_k=5, variance_percentage=0.95):
    """
    :param scaled_df: Spark DataFrame
    :param init_k: Integer initial value for PCA hyper parameters k
    :param variance_percentage: Double expected minimum feature variance explained via PCA features
    """
    pca = PCA(k=init_k, inputCol="features", outputCol="pcaFeatures")
    pca_model = pca.fit(scaled_df)
    for i in range(10):
        if pca_model.explainedVariance.sum() >= variance_percentage:
            print("\n the first {} features of PCA model represents {}% of the data".format(init_k + i,
                                                                                            variance_percentage))
            break
        pca = PCA(k=init_k + i, inputCol="features", outputCol="pcaFeatures")
        pca_model = pca.fit(scaled_df)

    pca_results = pca_model.transform(scaled_df).select("userId", "label", "pcaFeatures")
    return pca_results


if __name__ == '__main__':
    # create spark session and load dataset
    spark = SparkSession.builder.appName("Udacity Sparkify Project").getOrCreate()
    spark.conf.set("spark.sql.crossJoin.enabled", "true")
    data_path = 'mini_sparkify_event_data.json'
    df_raw = spark.read.json(data_path)
    df_raw.describe().show()

    df_clean = cleanse_numericals_and_categoricals(df_raw)

    df_labelled = generate_churn_label(df_clean)
    df_raw = None
    df_clean = None
    df = generate_page_based_features(df_labelled)
    df = generate_dummies_based_features(df)
    df = generate_session_based_features(df)

    df = df.drop("artist", "length", "firstName", "lastName", "sessionId", "location", "method", "userAgent", "page",
                 "song", "auth", "registration", "ts", "date", "day", "month", "hour", "monthDay", "status", "weekDay")
    df = df.dropDuplicates()
    df.printSchema()

    df = df.toDF("userId", "avgSessionsMonth", "avgSessionMonthDuration", "avgSessionitemsMonth", "avgSessionsDay",
                 "avgSessionDayDuration", "avgSessionitemsDay", "activeDuration",
                 "userDeviceLinux", "userDeviceMachintosh", "userDeviceWindows", "userDeviceiPad", "userDeviceiPhone",
                 "levelfree", "levelpaid",
                 "genderF", "genderM", "AboutPageMonth", "AddFriendPageMonth", "AddtoPlaylistPageMonth",
                 "DowngradePageMonth", "ErrorPageMonth", "HelpPageMonth",
                 "HomePageMonth", "LogoutPageMonth", "NextSongPageMonth", "RollAdvertPageMonth",
                 "SaveSettingsPageMonth", "SettingsPageMonth", "SubmitDowngradePageMonth",
                 "SubmitUpgradePageMonth", "ThumbsDownPageMonth", "ThumbsUpPageMonth", "UpgradePageMonth",
                 "itemInSession", "avgLength", "churn")
    scaled_df = feature_scaling(df)

    ratio = 0.8
    train_scaled_df = scaled_df.sampleBy('label', fractions={0.0: ratio, 1.0: ratio}, seed=123)
    test_scaled_df = scaled_df.subtract(train_scaled_df)

    plot_correlation(scaled_df, "features")

    for clf_name in ["LogisticRegression", "GBTClassifier", "RandomForestClassifier"]:
        print("\n using scaled features train, test, and visualized results of: {}".format(clf_name))
        train_predict_models(train_scaled_df, test_scaled_df, clf_name)

    ratio = 0.8
    pca_result = find_best_pca_param(scaled_df, init_k=12, variance_percentage=0.95)
    train_pca_df = pca_result.sampleBy('label', fractions={0.0: ratio, 1.0: ratio}, seed=123)
    test_pca_df = pca_result.subtract(train_pca_df)

    for clf_name in ["LogisticRegression", "GBTClassifier", "RandomForestClassifier"]:
        print("\n using PCA features train, test, and visualized results of: {}".format(clf_name))
        train_predict_models(train_pca_df, test_pca_df, clf_name, featuresCol="pcaFeatures", labelCol="label")
