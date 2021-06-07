
## creating a spark session
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.sql.functions import mean,col,split, col, regexp_extract, when, lit
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import QuantileDiscretizer
from pyspark.ml.classification import LogisticRegression


spark = SparkSession.builder.appName('titanic-trainig-example').getOrCreate()

# This function use to print feature with null values and null count 
def null_value_count(df):
  null_columns_counts = []
  numRows = df.count()
  for k in df.columns:
    nullRows = df.where(col(k).isNull()).count()
    if(nullRows > 0):
      temp = k,nullRows
      null_columns_counts.append(temp)
  return(null_columns_counts)

#------------------------------------------------------------

if __name__ == "__main__":
    
    train = spark.read.csv('./data/train.csv', header = True, inferSchema=True)
    test = spark.read.csv('./data/test.csv', header = True, inferSchema=True)
    train.show()

    titanic_df = spark.read.csv('./data/train.csv', header = True, inferSchema=True)
    titanic_df.groupBy("Survived").count().show()
    gropuBy_output = titanic_df.groupBy("Survived").count()

    titanic_df.groupBy("Sex","Survived").count().show()

    titanic_df.groupBy("Pclass","Survived").count().show()

    # Calling function
    null_columns_count_list = null_value_count(titanic_df)
    spark.createDataFrame(null_columns_count_list, ['Column_With_Null_Value', 'Null_Values_Count']).show()

    mean_age = titanic_df.select(mean('Age')).collect()[0][0]
    print(mean_age)

    titanic_df.select("Name").show()

    titanic_df = titanic_df.withColumn("Initial",regexp_extract(col("Name"),"([A-Za-z]+)\.",1))
    titanic_df.show()
    titanic_df.select("Initial").distinct().show()

    titanic_df = titanic_df.replace(['Mlle','Mme', 'Ms', 'Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],
               ['Miss','Miss','Miss','Mr','Mr',  'Mrs',  'Mrs',  'Other',  'Other','Other','Mr','Mr','Mr'])

    titanic_df.select("Initial").distinct().show()
    titanic_df.groupby('Initial').avg('Age').collect()

    #Fixing nulls in age
    titanic_df = titanic_df.withColumn("Age",when((titanic_df["Initial"] == "Miss") & (titanic_df["Age"].isNull()), 22).otherwise(titanic_df["Age"]))
    titanic_df = titanic_df.withColumn("Age",when((titanic_df["Initial"] == "Other") & (titanic_df["Age"].isNull()), 46).otherwise(titanic_df["Age"]))
    titanic_df = titanic_df.withColumn("Age",when((titanic_df["Initial"] == "Master") & (titanic_df["Age"].isNull()), 5).otherwise(titanic_df["Age"]))
    titanic_df = titanic_df.withColumn("Age",when((titanic_df["Initial"] == "Mr") & (titanic_df["Age"].isNull()), 33).otherwise(titanic_df["Age"]))
    titanic_df = titanic_df.withColumn("Age",when((titanic_df["Initial"] == "Mrs") & (titanic_df["Age"].isNull()), 36).otherwise(titanic_df["Age"]))

    titanic_df.filter(titanic_df.Age==46).select("Initial").show()
    
    titanic_df.select("Age").show()
    titanic_df = titanic_df.na.fill({"Embarked" : 'S'})
    titanic_df = titanic_df.drop("Cabin")
    titanic_df = titanic_df.withColumn("Family_Size",col('SibSp')+col('Parch'))
    titanic_df.groupBy("Family_Size").count().show()
    titanic_df = titanic_df.withColumn('Alone',lit(0))
    titanic_df = titanic_df.withColumn("Alone",when(titanic_df["Family_Size"] == 0, 1).otherwise(titanic_df["Alone"]))

    #Converting sex columns into numbers
    indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(titanic_df) for column in ["Sex","Embarked","Initial"]]
    pipeline = Pipeline(stages=indexers)
    titanic_df = pipeline.fit(titanic_df).transform(titanic_df)

    #Drop columns which are not required
    titanic_df = titanic_df.drop("PassengerId","Name","Ticket","Cabin","Embarked","Sex","Initial")
    titanic_df.show()

    # Feature vector
    feature = VectorAssembler(inputCols=titanic_df.columns[1:],outputCol="features")
    feature_vector= feature.transform(titanic_df)

    feature_vector.show()

    #Splitting data into training and test
    (trainingData, testData) = feature_vector.randomSplit([0.8, 0.2],seed = 11)
    

    #------- Training using LogisticRegresion
    lr = LogisticRegression(labelCol="Survived", featuresCol="features")
    #Training algo
    lrModel = lr.fit(trainingData)
    lr_prediction = lrModel.transform(testData)
    lr_prediction.select("prediction", "Survived", "features").show()
    evaluator = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="accuracy")

    ## Evaluating training
    lr_accuracy = evaluator.evaluate(lr_prediction)
    print("Accuracy of LogisticRegression is = %g"% (lr_accuracy))
    print("Test Error of LogisticRegression = %g " % (1.0 - lr_accuracy))
    
    
    
