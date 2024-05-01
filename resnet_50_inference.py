#from 
#https://docs.aws.amazon.com/sagemaker/latest/dg/adapt-inference-container.html
#https://knowledge.udacity.com/questions/775344#775906
#https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#load-a-model
# Based on tutorial from: https://sagemaker-examples.readthedocs.io/en/latest/frameworks/pytorch/get_started_mnist_deploy.html



import json, logging, sys, os, io, requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

JPEG_CONTENT_TYPE = 'image/jpeg'
JSON_CONTENT_TYPE = 'application/json'
ACCEPTED_CONTENT_TYPE = [ JPEG_CONTENT_TYPE ] #Add support for jpeg images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


def Net():
    '''
    Instantitate a pre trained resnet 50. This is the same code as used in the other files.
    '''
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features
    model.fc = nn.Sequential(
                   nn.Linear(2048, 128),
                   nn.ReLU(inplace=True),
                   nn.Linear(128, 133))
    
    return model


'''
Loads a model artifact and returns a model object for generating predictions
'''  
def model_fn(model_dir):
    print("In model_fn. Model directory is -")
    print(model_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    '''
    Load model state from directory
    '''    
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        print("Loading the dog-classifier model")
        checkpoint = torch.load(f , map_location =device)
        model.load_state_dict(checkpoint)
        print('MODEL-LOADED')
        print('Model loaded successfully')
    model.eval()
    
    return model


'''
This function handles data decoding. 
'''  
def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    print('Deserializing the input data.')
    print(f'Request body CONTENT-TYPE is: {content_type}')
    print(f'Request body TYPE is: {type(request_body)}')
    
    if content_type == JPEG_CONTENT_TYPE: 
        print('Loaded JPEG content')
        return Image.open(io.BytesIO(request_body))
    
    # process a URL submitted to the endpoint
    if content_type == JSON_CONTENT_TYPE:
        print('Loaded JSON content')
        print(f'Request body is: {request_body}')
        request = json.loads(request_body)
        print(f'Loaded JSON object: {request}')
        url = request['url']
        img_content = requests.get(url).content
        return Image.open(io.BytesIO(img_content))
    
    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))

'''
Take in the input object, transform to the required format for the model and returns a prediction
''' 
def predict_fn(input_object, model):
    print('In predict fn')
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()])
    
    print("Transforming input")
    input_object=test_transform(input_object)
    
    with torch.no_grad():
        print("Calling model")
        prediction = model(input_object.unsqueeze(0))
    return prediction

'''
Process and output the model predictions
''' 
def output_fn(predictions, content_type):
    print(f'Postprocess CONTENT-TYPE is: {content_type}')
    assert content_type == JSON_CONTENT_TYPE
    res = predictions.cpu().numpy().tolist()
    return json.dumps(res)