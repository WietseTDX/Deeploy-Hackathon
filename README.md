# Diabetes patients readmission prediction
This repository contains an example model for predicting whether a diabetes patient will be readmitted to the hospital. 
It uses an open dataset from the years 1999-2008 that contains 101766 records. 
For more information, check out the [UCI webpage](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008).

## Model training
A trained XGBoost model is already available at `model/model.bst`. 
If you want to retrain the model or build an explainer, you need to install the `dev-requirements.txt` using pip or poetry. 
Then, download the dataset by running `python download_dataset.py` in the `data` folder. 
The notebook `notebooks/train_diabetes_model.py` contains code to preprocess the data and train the model. 

## Deploy to Deeploy
You can fork this repository or copy the `model` folder and `metadata.json` to a new repository. 
During model creation, specify the XGBoost model framework and for the purpose of the hackathon disable serverless.
If you add an explainer, update this repository with the explainer object in the `explainer` folder, or add a `reference.json` with a link to a custom Docker image.

## Additional details
The preprocessing steps include the removal of some unimportant features and some rows that lack data. This results in 71 features. When working on explainability, think about the encoded features and if it makes sense to include those in the explainer. Alternatively you can use the raw input.
For testing your model, you can use the example input as stated in the metadata or copy one of the rows in the test set (see notebook).

## Monitoring limitations
When you have deployed this model to Deeploy, it will output the probability of readmission. To fully make use of Deeploy's monitoring capabilities, use a binary output. With binary output we can calculate accuracy, f1-score and even compare results with human evaluations.
There are two ways of outputting binary predictions for this model:  
* Use a custom Docker for the model. With a custom Docker you could load in the XGBoost model as XGBClassifier, or round the output after `model.predict()` before returning the result. You can use the [deeploy cli](https://docs.deeploy.ml/next/python-client/cli) to easily create a custom Docker. 
* Use a transformer. With a transformer you can preprocess or postprocess data. This transformer is also a custom Docker container that transforms the predictions by rounding them. It can also be easily created using the [deeploy cli](https://docs.deeploy.ml/next/python-client/cli).