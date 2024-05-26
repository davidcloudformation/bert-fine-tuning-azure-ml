# BERT Fine-Tuning for Text Classification on Azure ML

This repository contains code for fine-tuning a BERT model for text classification using Azure Machine Learning.

## Project Structure

```
bert-fine-tuning-azure-ml/
│
├── data/
│   └── train.csv
│   └── test.csv
│
├── scripts/
│   └── train.py
│   └── score.py
│
├── environment.yml
├── config.json
├── README.md
└── .gitignore
```

## Setup

1. **Clone the repository**:
   ```sh
   git clone https://github.com/davidcloudformation/bert-fine-tuning-azure-ml.git
   cd bert-fine-tuning-azure-ml
   ```

2. **Create Azure ML Workspace**:
    - Follow the instructions to create an Azure ML workspace from the [Azure Portal](https://portal.azure.com/).

3. **Install Azure ML SDK**:
   ```sh
   pip install azureml-core azureml-sdk azureml-widgets
   ```

4. **Configure Workspace**:
    - Create a `config.json` file with your Azure ML workspace details.

## Training

1. **Upload Data**:
    - Upload your training and test data to the `data/` directory.

2. **Run Training Script**:
   ```sh
   python scripts/train.py
   ```

## Deployment

1. **Deploy Model**:
    - Use the `score.py` script to deploy the model as a web service on Azure ML.

## Inference

1. **Run Inference**:
    - Send a POST request to the deployed web service with the input data to get predictions.

## License

This project is licensed under the MIT License.



