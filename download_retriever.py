import os
from dotenv import load_dotenv
from comet_ml import API

load_dotenv()

# Initialize the Comet API
api = API()

# Get the model object from the Model Registry
# Replace 'YOUR_WORKSPACE', 'YOUR_MODEL_NAME', and 'YOUR_VERSION' with your actual values
model = api.get_model(os.getenv("COMET_WORKSPACE"), "retriever_v2")

# Download the specific version of the model
# The files will be saved in the 'output_folder'
model.download(version="1.0.0", output_folder="./models")