# pyspark-titanic-training

This repo is an example PySpark model that is conformed for use with ModelOp Center and the ModelOp Spark Runtime Service.

## Assets

There are three assets that are used to run this example:

| Asset Type | Repo File | HDFS Path | Description |
| --- | --- | --- | --- |
| Model source code | `titanic.py` | - | Pyspark code to train a model with `LogisticRegression` using the titanic dataset. |
| Input Asset1 | `train.csv` | `/hadoop/demo/pyspark-titanic-training/data/` | Input file for the model `fit()` function. The HDFS path can vary based on the `external_inputs` param of the `train()` function  |
| Input Asset2 | `test.csv` | `/hadoop/demo/pyspark-titanic-training/data/` | Test data. The HDFS path can vary based on the `external_inputs` param of the `train()` function  |
| Output Asset | `output.model` | `/hadoop/demo/pyspark-titanic-training/` | Output file from the model `train()` function. The HDFS path can vary based on the `external_outputs` param of the `train()` function  |


## Mocaasin Tests

1. Verify that the input asset 1 `train.csv` and input asset 2 `test.csv` (above) exists at `/hadoop/demo/pyspark-titanic-training/data/` in the Spark cluster HDFS. The input asset can be in a different location, but you must update the input asset URL in testing steps below.
2. Import this repository to ModelOp Center, selecting external repository as HDFS.

### Test Training Job
1. Create a new training job with the following HDFS URL assets:
   
    JOB inputs:
    - Input asset 1: `hdfs:///hadoop/demo/pyspark-titanic-training/data/train.csv`
    - Input asset 2: `hdfs:///hadoop/demo/pyspark-titanic-training/data/test.csv`
    
   JOB outputs:
    - Output asset: `hdfs:///hadoop/demo/pyspark-titanic-training/output.model`


2. Select the Spark-Runtime engine, and launch the job
3. Wait for the job to enter the `COMPLETE` state
4. Inside the Spark cluster, try: `hadoop fs -ls /hadoop/demo/pyspark-titanic-training/output.model` , you should be able to see files and folders.


Notes:
1. PySpark code expects to receive the input assets in the order described above, first `/train.csv` and then `/test.csv`.
2. Once the `training` job finishes successfully, in order to run again the same job, output must be removed or output asset location should be changed, otherwise subsequent jobs will fail.