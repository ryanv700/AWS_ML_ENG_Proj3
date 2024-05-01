# AWS_ML_ENG_Proj3
AWS ML Engineer Nanodegree project: "Image Classification using AWS Sagemaker"

# Dog Images Classification Project

## Introduction
This project aims to classify dog images for a large variety of dog breeds by finetuning a pretrained ResNet50 Computer Vision Model. The model is trained on a dataset of dog images and can predict the breed of a given dog image.

## Project Setup Instructions
1. Clone the project repository from GitHub.
2. Download the train_and_deploy.ipynb notebook, hop.py, and train_model.py
3. Download the dataset of dog images from the link found in the jupyter notebook.
4. Run the code in the notebook which will: download the data, store in Amazon S3, Fetch the data from S3 and use the hpo.py script to run a sagemaker training job
5. Train the model in the notebook using the estimator and the train_model.py script
6. Deploy the model to an endpoint using AWS SageMaker by running the notebook

## Files Explanation
- `hpo.py`: using SageMaker script mode to run a hyperparameter tuning job to fine tune the Resnet 50 on the dogimages data.
- `train_model.py`: Trains the final model uses the best hyparameters on the full training data set.
- `resnet_50_inference.py`: Deploys the trained model to an endpoint using AWS SageMaker. Uses predict and output functions to serve inferences

## Code Sample for Querying a Model Endpoint
```python
import boto3

test_image = "./dogImages/test/021.Belgian_sheepdog/Belgian_sheepdog_01555.jpg"
with open(test_image, "rb") as image:
    f = image.read()
    img_bytes = bytearray(f)

type(img_bytes)

from PIL import Image
import io
Image.open(io.BytesIO(img_bytes))

response=predictor.predict(img_bytes, initial_args={"ContentType": "image/jpeg"})
```

Screenshots

This shows what your completed training jobs should look like in the SageMaker Studio Portal

![completed_training_jobs](https://github.com/ryanv700/AWS_ML_ENG_Proj3/assets/56355045/805ac99d-2ccb-46a9-a7f3-8b0e58ec3d6e)

This is what you should see in the Endpoints after deploying your model

![AWS_ML_ENG_Proj_3_model_endpoint_Screenshot](https://github.com/ryanv700/AWS_ML_ENG_Proj3/assets/56355045/d85404ed-f9e3-4caf-995b-86d173a3cee2)

