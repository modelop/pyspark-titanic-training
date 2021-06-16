from __future__ import print_function

## creating a spark session
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.sql.functions import mean,col,split, col, regexp_extract, when, lit
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import QuantileDiscretizer
from pyspark.ml.classification import LogisticRegression
from typing import List

# modelop.init
def init():
  print("Begin function...", flush=True)


# modelop.train
def train(external_inputs: List, external_outputs: List, external_model_assets: List):

  global SPARK
  spark = SparkSession.builder.appName('modelop-titanic-pyspark-trainig').getOrCreate()
  print("Spark variable:", spark, flush=True)


  input_train_asset_path, input_test_asset_path = parse_input_assets(external_inputs)

  #train = spark.read.csv(input_train_asset_path, header = True, inferSchema=True)
  #test = spark.read.csv(input_test_asset_path, header = True, inferSchema=True)
  #train.show()

  #titanic_df = spark.read.csv(input_train_asset_path, header = True, inferSchema=True)
  titanic_df = spark.read.format("csv").option("header", "true").option("inferSchema","true").load(input_train_asset_path)
  titanic_df.printSchema()

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




def null_value_count(df):
  """ count the null values """
  null_columns_counts = []
  numRows = df.count()
  for k in df.columns:
    nullRows = df.where(col(k).isNull()).count()
    if(nullRows > 0):
      temp = k,nullRows
      null_columns_counts.append(temp)
  return(null_columns_counts)



def parse_input_assets(external_inputs: List):
    """Returns a tuple (input train asset hdfs path, input test asset hdfs path)"""

    # Fail if more assets than expected
    if len(external_inputs) != 2:
        raise ValueError("Only two input asset should be provided")

    # There's only one key-value pair in each dict, so
    # grab the first value from both
    input_train_asset = external_inputs[0]
    input_test_asset = external_inputs[1]

    # Fail if assets are JSON
    if ("fileFormat" in input_train_asset) and (input_train_asset["fileFormat"] == "JSON"):
        raise ValueError("Input file traning format is set as JSON but must be CSV")
      
    # Fail if assets are JSON
    if ("fileFormat" in input_test_asset) and (input_test_asset["fileFormat"] == "JSON"):
        raise ValueError("Input file test format is set as JSON but must be CSV")

    # Return paths from file URLs
    input_train_asset_path = input_train_asset["fileUrl"]
    input_tests_asset_path = input_test_asset["fileUrl"]

    return (input_train_asset_path, input_tests_asset_path)


#------------------------------------------------------------
# For local testing direcly with Spark
#if __name__ == "__main__":
  

#  input_data_json = [{"fileFormat":"CSV", "fileUrl":"/titanic/data/train.csv"},{"fileFormat":"CSV", "fileUrl":"/titanic/data/test.csv"}]
#  output_list = []
#  external_assets_list = []
  
#  train(input_data_json, output_list,external_assets_list)

  
