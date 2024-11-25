
# ml-framework-k8s

# ML Framework - Training and Docker Testing Guide

Welcome to the ML Framework! This guide provides step-by-step instructions on how to train a machine learning model and how to test the trained model using Docker. Whether you're a developer or a data scientist, this guide will help you get started with training your models and deploying them efficiently.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Project Structure](#project-structure)
3. [1. Training a Model](#1-training-a-model)
    - [Step 1: Setup the Environment](#step-1-setup-the-environment)
    - [Step 2: Configure the Training](#step-2-configure-the-training)
    - [Step 3: Prepare the Data](#step-3-prepare-the-data)
    - [Step 4: Train the Model](#step-4-train-the-model)
    - [Step 5: Verify the Model Artifacts](#step-5-verify-the-model-artifacts)
4. [2. Testing on Docker](#2-testing-on-docker)
    - [Step 1: Build the Docker Image](#step-1-build-the-docker-image)
    - [Step 2: Run the Docker Container](#step-2-run-the-docker-container)
    - [Step 3: Test the API Endpoints](#step-3-test-the-api-endpoints)
    - [Step 4: Debugging and Logs](#step-4-debugging-and-logs)
5. [Additional Resources](#additional-resources)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before you begin, ensure you have the following installed on your system:

- **Python 3.8+**
- **Docker**
- **Git** (optional, for version control)
- **Virtual Environment Tool** (`venv` or `conda`)

Additionally, make sure you have access to the project's repository and necessary data files.

---

## Project Structure

Here's the project structure tree
```bash

ml-framework-k8s/
├── artifacts/
│   └── (some examples saved here)
├── api/
│   ├── app.py
│   └── __init__.py
├── data/
│   ├── customers.csv
│   ├── noncustomers.csv
│   ├── actions.csv
│   └── loader.py
├── models/
│   ├── __init__.py
│   ├── base_model.py
│   ├── registry.py
│   ├── sklearn_logistic_model.py
│   ├── tensorflow_model.py
│   └── huggingface_model.py
├── preprocess/
│   ├── __init__.py
│   ├── base_preprocessor.py
│   ├── registry.py
│   ├── numeric_preprocessor.py
│   ├── huggingface_preprocessor.py
│   └── tree_preprocessor.py
├── model_training.py
├── config.yaml
├── Dockerfile
├── requirements.txt
└── README.md
```
---

## 1. Training a Model

Training a model involves preparing your environment, configuring the training parameters, preparing the data, executing the training script, and verifying the trained model.

### Step 1: Setup the Environment

1. **Clone the Repository**

   If you haven't already, clone the project repository:

   ```bash
   git clone https://github.com/yourusername/ml-framework-k8s.git
   cd ml-framework-k8s
   ```
2. **Install Dependencies**
   ```bash 
   pip install -r requirements.txt

   ```
3. **Prepares the data**
   Organize Data Files
   Ensure that your data files (customers.csv, noncustomers.csv, actions.csv) are placed in the data/ directory.

   Understand the DataLoader

   The data/loader.py script is responsible for loading and preprocessing the data. Ensure that it correctly reads your CSV files and combines them as needed. (Currently:
   1. It is set to take average action metrics up to the date before conversion.
   2. Dummy Variables are used to process categorical variables. )

4. **Training the model**
    ```bash
    python model_training.py --config config.yaml
    ```
    Output:

    The script will print logs indicating the progress through data loading, preprocessing, training, evaluation, and saving the model.
5. **Verify the Model Artifacts**
	Check the artifacts/ Directory
    After successful training, the trained model file should be saved in the artifacts/ directory. For example:

    Logistic Regression: artifacts/logistic_regression_model.pkl
    TensorFlow: artifacts/tensorflow_model.h5
    Hugging Face: artifacts/huggingface_model/

## 2. Testing On Docker
3. **Build the Docker image**
	```bash 
	docker build -t ml-framework:latest .
	```

4. **(Optional) Deploy to Kubernetes** 


6. **Run the Docker Container**

	Run the Docker container, mapping the container's port to a port on   your host machine:
    ```bash
    docker run --privileged -p 8000:8000 ml-framework:latest
    ```
7. **Test the API Endpoints** 

	Calling the predict endpoint with the company id for results:
    ```bash

    curl -X POST "http://localhost:8000/predict/" \
	-H "Content-Type: application/json" \
	-d '{"ids": [199, 147]}'
	``` 
## Troubleshooting
1. Error: ModuleNotFoundError: No module named 'data'

Cause: Python cannot locate the data module due to incorrect project structure or Python path.

Solution:

Ensure data/ directory exists and contains an __init__.py file.
Verify you're running the script from the project root.

2. Insufficient Resources

Cause: Docker container lacks sufficient memory or CPU resources.

Solution:

Allocate more resources to Docker (Docker Desktop settings).
Limit the number of threads used by the model to reduce resource consumption.
docker run with the --privileged flag.