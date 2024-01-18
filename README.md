# News-Classification

Tutorial: https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html

This code shows the complete process of loading and preprocessing data, defining a text classification model, training the model, and evaluating its performance on test data. It also includes saving and loading the trained model, as well as making predictions on new text samples.

## Data Processing:
- Use TorchText to load the AG_NEWS dataset.
- Tokenize the text using the basic English tokenizer.
- Build a vocabulary from the training dataset.

## Data Collation Function:
- Define a function collate_batch to collate a batch of data.
- Convert labels and text to tensors and create offsets for the text.

## Model Definition:
- Define a text classification model (TextClassificationModel) using an embedding bag and a linear layer.
- Initialize the model's weights.

## Model Initialization and Training:
- Initialize an instance of the model, specifying the vocabulary size, embedding dimension, and number of classes.
- Define a training function (train) that trains the model using cross-entropy loss and stochastic gradient descent (SGD).
- The learning rate is adjusted using gradient clipping and a learning rate scheduler.

## Data Loading and Splitting:
- Load the training and test datasets and convert them to map-style datasets.
- Split the training dataset into training and validation sets.

## Training the Model:
- Train the model for a specified number of epochs.
- Evaluate the model on the validation set and adjust the learning rate based on validation accuracy.

## Save and Load Model:
- Save the trained model's weights to a file (news_classification_model.pth).
- Load the saved model for evaluation.

## Evaluate the Model on Test Data:
Evaluate the loaded model on the test dataset and print the test accuracy.

## Predict on a Sample News:
- Define a function (predict) to make predictions on a given text.
- Provide a sample news text and predict its category.
