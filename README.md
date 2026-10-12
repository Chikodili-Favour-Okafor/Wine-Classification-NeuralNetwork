# Wine Classification Using PyTorch

## Project Overview

This project implements a multiclass classification neural network using PyTorch to predict wine classes from their chemical properties. It uses the Wine recognition dataset provided by Scikit-learn and demonstrates the process of preparing data, building a neural network, training the model, validating its performance, and evaluating it on unseen test data.

The project is a practical introduction to deep learning and supervised machine learning using PyTorch.

## Objectives

* Explore and prepare a classification dataset.
* Split data into training, validation, and testing sets.
* Standardize numerical features.
* Build and train a feedforward neural network using PyTorch.
* Monitor validation loss and select the best-performing model.
* Evaluate model performance using classification metrics.
* Save the trained model weights and fitted scaler for future predictions.

## Dataset

The project uses the built-in Wine recognition dataset from `sklearn.datasets.load_wine()`.

* **Samples:** 178
* **Input features:** 13
* **Target classes:** 3
* **Task:** Multiclass classification

The dataset contains numerical measurements of wine chemical properties, including alcohol, malic acid, ash, magnesium, and flavonoids. Each sample belongs to one of three wine classes.

## Technologies and Libraries

* Python
* PyTorch
* Scikit-learn
* NumPy
* Pandas
* Matplotlib
* Joblib
* Jupyter Notebook

## Project Workflow

1. **Data Loading and Exploration:** Load the Wine dataset and inspect its dimensions, features, target classes, and data quality.
2. **Data Preparation:** Separate the input features from the target labels.
3. **Train-Validation-Test Split:** Split the dataset into 70% training, 15% validation, and 15% testing data using stratified sampling to preserve class proportions.
4. **Feature Scaling:** Use `StandardScaler` to standardize the features. Fit the scaler on the training data and use it to transform the training, validation, and testing sets.
5. **Tensor Conversion:** Convert features and target labels into PyTorch tensors with appropriate data types.
6. **Data Loading:** Use PyTorch `DataLoader` to feed training samples to the network in batches.
7. **Neural Network Construction:** Build a feedforward neural network with hidden layers, ReLU activation functions, and dropout.
8. **Model Training:** Train the network using cross-entropy loss, the Adam optimizer, and backpropagation.
9. **Validation and Model Selection:** Monitor validation loss after each epoch and retain the model state with the lowest validation loss.
10. **Model Evaluation:** Evaluate the selected model on the held-out test set using accuracy, precision, recall, F1-score, and a confusion matrix.
11. **Model Saving:** Save the trained model weights and fitted scaler for future use.

## Neural Network Architecture

| Layer          | Configuration    |
| -------------- | ---------------- |
| Input layer    | 13 features      |
| Hidden layer 1 | 32 neurons, ReLU |
| Dropout        | 20%              |
| Hidden layer 2 | 16 neurons, ReLU |
| Dropout        | 20%              |
| Output layer   | 3 neurons        |

The output layer produces three logits, one for each wine class. `CrossEntropyLoss` is used to train the model for multiclass classification.

## Training Configuration

| Parameter       | Value              |
| --------------- | ------------------ |
| Training set    | 70%                |
| Validation set  | 15%                |
| Test set        | 15%                |
| Random state    | 42                 |
| Batch size      | 16                 |
| Training epochs | 100                |
| Optimizer       | Adam               |
| Learning rate   | 0.001              |
| Loss function   | Cross-entropy loss |
| Dropout rate    | 0.2                |

## Model Performance

The model achieved approximately **96.3% test accuracy** on the held-out test set.

### Classification Report

| Class            | Precision | Recall | F1-score | Support |
| ---------------- | --------: | -----: | -------: | ------: |
| Class 0          |      1.00 |   1.00 |     1.00 |       9 |
| Class 1          |      0.92 |   1.00 |     0.96 |      11 |
| Class 2          |      1.00 |   0.86 |     0.92 |       7 |
| Macro average    |      0.97 |   0.95 |     0.96 |      27 |
| Weighted average |      0.97 |   0.96 |     0.96 |      27 |

### Confusion Matrix

Rows represent actual classes, while columns represent predicted classes.

```text
[[ 9  0  0]
 [ 0 11  0]
 [ 0  1  6]]
```

The model correctly classified 26 of the 27 test samples. One sample from Class 2 was incorrectly classified as Class 1.

These results demonstrate strong performance on this particular test split. However, the test set is small, so performance may vary on other unseen samples.

## Project Structure

```text
Wine-Classification/
├── notebook/
│   └── wine_classification.ipynb
├── models/
│   ├── wine_model.pth
│   └── wine_scaler.pkl
├── README.md
└── requirements.txt
```

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Wine-Classification-NeuralNetwork.git
cd Wine-Classification-NeuralNetwork
```

Note: Replace `YOUR_USERNAME` with the GitHub username associated with the repository.

### 2. Install the dependencies

```bash
pip install -r requirements.txt
```

### 3. Open the notebook

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open `notebook/WineClassDetactor.ipynb` and run the cells in order.

## Saved Model Artifacts

* `models/wine_model.pth`: Contains the trained PyTorch model weights.
* `models/wine_scaler.pkl`: Contains the fitted feature scaler.

To use the saved model for predictions, recreate the same neural network architecture, load its state dictionary, and apply the saved scaler to new input features before passing them to the model.

## Key Learnings

This project provided practical experience with:

* Multiclass classification using neural networks.
* PyTorch tensors, datasets, and data loaders.
* Forward propagation and backpropagation.
* Cross-entropy loss and the Adam optimizer.
* Feature scaling and data leakage prevention.
* Validation-based model selection.
* Classification metrics and confusion matrix interpretation.
* Saving model weights and preprocessing artifacts.

## Future Improvements

* Compare the neural network with Logistic Regression, Random Forest, and Support Vector Machines.
* Experiment with different network architectures and hyperparameters.
* Build a simple interface for predicting wine classes from input features.

## Author

**Chikodili Favour Okafor**

Data Scientist | AI / Machine Learning Enthusiast

[GitHub Profile](https://github.com/chikodili-Favour-Okafor)
