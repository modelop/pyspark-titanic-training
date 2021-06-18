# pyspark-titanic-training

This repo is an example PySpark model that is conformed for use with ModelOp Center and the ModelOp Spark Runtime Service.

## Assets

There are three assets that are used to run this example:

| Asset Type | Repo File | HDFS Path | Description |
| --- | --- | --- | --- |
| Model source code | `titanic.py` | `/hadoop/demo/titanic-spark/titanic` | Spark model binary compressed as a zip file in this repo, but must be expanded and be available in the Spark cluster HDFS for the model's `init()` function to run |
| Input Asset | `test.csv` | `/hadoop/demo/titanic-spark/test.csv` | Input file for the model `score()` function. The HDFS path can vary based on the `external_inputs` param of the `score()` function  |
| Output Asset | `titanic_output.csv` | `/hadoop/demo/titanic-spark/titanic_output.csv` | Output file from the model `score()` function. The HDFS path can vary based on the `external_outputs` param of the `score()` function  |
