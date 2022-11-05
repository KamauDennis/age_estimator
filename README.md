# Training and Deploying an Age Estimation model

The aim of this project is to train,save,deploy and interact with an age detection model in production environments.To serve the model,we'll use TensorFlow Serving which is a high performance serving system for machine learning models.

## Requirements

First of all,you need to make sure that python 3 is installed then use pip to install the following libraries;

- numpy
- tensorflow

Also,download [UTKFACE](https://drive.google.com/file/d/1hP2QveO67LHciaUNRleh1NtpUGYOMFD0/view?usp=sharing) dataset. 

For deployment,you need to make sure TF Servng is installed.There are many ways of installing TF Serving:using a Docker image,using the system's package manager,installing from source,and more.You can use the Docker option,which is highly recomended by TensorFlow team.You first need to install [Docker](https://docker.com).Then download the official TF Serving docker image.

`docker pull tensorflow/serving`

Finally,for the client side of the deployment you need to install the python package `tensorflow-serving-api`,in case you want to use the gRPC API,which is faster than the REST API regarding the latency and inference time.

`pip install tensorflow-serving-api`

## Usage

You can look into the following notebook,[training an age detection model using TensorFlow](/Training%20an%20Age%20Detection%20model%20using%20TensorFlow.ipynb) to check the process used to train the model.

To train the model run the following command:

`python training.py`

## Deployment
After running the training script an age detector model is saved,which we'll serve using TF Serving.If you don't have computer resources to train the model you can download my [age_detector_model](https://drive.google.com/drive/folders/17h_1o9H3rCf5IOVvJK4WnJrPv-E9pkL4?usp=sharing) that I developed.

Once you download the official TF Serving Docker image,you can create a Docker container to run this image:

`export ML_PATH="$HOME/age_detector"`point to this project,wherever it is

``sudo docker run -it --rm -p 8500:8500 -p 8501:8501 \
                -v "$ML_PATH/age_detector_model:/models/age_detector_model" \
                -e MODEL_NAME=age_detector_model \
                tensorflow/serving``

That's it!TF Serving is running.It loaded the age_detector_model and it is serving it through both gRPC(on port 8500) and REST(on port 8501).To see the process of querying the server check the following [TFServing notebook](TFServing.ipynb).
