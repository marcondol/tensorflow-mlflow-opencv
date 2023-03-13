FROM tensorflow/tensorflow:latest

# Install OpenCV
RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev
RUN pip install opencv-python-headless

# Install MLflow
RUN pip install mlflow

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app
# Set the entrypoint to run mlflow server
EXPOSE 5000
ENTRYPOINT ["mlflow", "server", "--host", "0.0.0.0"]
