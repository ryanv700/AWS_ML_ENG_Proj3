# AWS_ML_ENG_Proj3
AWS ML Engineer Nanodegree project: "Image Classification using AWS Sagemaker"

# Dog Images Classification Project

## Introduction
This project aims to classify dog images using machine learning techniques. The model is trained on a dataset of dog images and can predict the breed of a given dog image.

## Project Setup Instructions
1. Clone the project repository from GitHub.
2. Install the required dependencies by running `pip install -r requirements.txt`.
3. Download the dataset of dog images from [link to dataset].
4. Preprocess the dataset by running `python preprocess.py`.
5. Train the model by running `python train.py`.
6. Deploy the model to an endpoint using AWS SageMaker by running `python deploy.py`.

## Files Explanation
- `preprocess.py`: Preprocesses the dataset of dog images.
- `train.py`: Trains the machine learning model on the preprocessed dataset.
- `deploy.py`: Deploys the trained model to an endpoint using AWS SageMaker.
- `query.py`: Provides a code sample for querying the model endpoint and getting predictions.

## Code Sample for Querying a Model Endpoint
```python
import boto3

# Create a SageMaker client
sagemaker_client = boto3.client('sagemaker')

# Create a runtime client
runtime_client = boto3.client('sagemaker-runtime')

# Specify the endpoint name
endpoint_name = 'your-endpoint-name'

# Specify the image file path
image_path = 'path/to/your/image.jpg'

# Read the image file
with open(image_path, 'rb') as f:
    image = f.read()

# Invoke the endpoint for prediction
response = runtime_client.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType='application/x-image',
    Body=image
)

# Get the prediction result
result = response['Body'].read().decode()
print(result)
Insights from the Model
The trained model achieved an accuracy of X% on the test dataset. It performed well in classifying different dog breeds, but struggled with images that contained multiple dogs or other objects.

Screenshots
