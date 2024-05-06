# Image Captioning Project

This project demonstrates an implementation of image captioning using deep learning techniques, combining computer vision with natural language processing to generate textual descriptions for images.

## Objective

The objective of this project is to build a model that can automatically generate captions for given images, using pre-trained convolutional neural networks (CNNs) for feature extraction and LSTM for sequence generation.

## Problem Statement

Image captioning involves the following issues:
- Extracting meaningful features from images using CNNs.
- Generating coherent and contextually relevant captions using RNNs.
- Evaluating the quality of generated captions using metrics like BLEU score.

## Methodology

**Importing Libraries**
Necessary libraries are imported such as numpy, pandas, TensorFlow, etc. These libraries perform tasks like data preparation, image processing, deep learning model building, and evaluation.

**Pre-Processing Captions**
To preprocess captions a function called text_preprocessing converts all characters to lowercase, removes non-alphabetic characters, removes extra whitespaces, and appends start and end tokens to each caption. This function is to prepare data for tokenization.

**Tokenization and Encoding**
The captions are tokenized into words using Tokenizer from Keras. It broke down the text into words and assigned a unique integer to each word. The maximum length of captions is calculated to prepare for padding sequences later.

**Image Feature Extraction**
A pre-trained convolutional neural network (DenseNet201) is used in code to extract features from images. The model is instantiated and the last layer is removed to obtain features from the images. Extracted features of all images are stored in a dictionary.

**Data Generation**
A class of CustomDataGenerator is defined to generate batches of data during model training. It takes image features, tokenized captions, and other parameters as input and gives batches of pairs of image captions for training.

**Modeling**
The architecture of the model is defined using the Functional API of Keras. It has two input layers: 
1)For image Features
2)For Tokenized Captions

The image features are passed through a dense layer, reshaped, and concatenated with the embedded captions sequences. For sequence processing the concatenated vector is fed as input into the LSTM layer. In the end, a dropout layer and dense layers process the output before predicting the next words.

There are a total of 10 layers

input1: Input layer for image features.
input2: Input layer for tokenized captions.
Dense: A dense layer for processing image features.
Reshape: Reshaping the output of the dense layer.
Embedding: Embedding layer for tokenized captions.
Concatenate: Concatenation layer to combine image features and embedded captions.
LSTM: Long Short-Term Memory (LSTM) layer for sequence processing.
Dropout: Dropout layers for regularization.
Add: 1 occurrence to add output of dense and dropout layer.
Dense: Dense layer for final prediction.

**Model Training**
Cross-entropy loss and Adam Optimizer are used to compile the model. For training and validation, Data generators are instantiated. Functions such as ModelCheckpoint, EarlyStopping, and ReduceLROnPlateau are used to adjust the learning rate during training by watching validation loss. The model is trained using the fit method.
**Caption Generation**
The trained model produces captions using the function predict_caption. It takes the image, tokenizer, maximum length, and image features as input and generates a caption using a greedy search strategy.

**Caption Prediction**
Randomly some images are selected and their captions are predicted. The predicted captions along with the corresponding images are displayed using the display_images function.

## Results

The trained model generates captions for images. The evaluation process includes calculating BLEU scores to measure the similarity between generated captions and reference captions.

## Usage

1. **Environment Setup**:
   - Ensure you have Python installed along with necessary libraries (TensorFlow, Keras, NLTK, etc.).
   - Download and install required packages using `pip` or `conda`.

2. **Data Preparation**:
   - Download and preprocess the image dataset (e.g., Flickr8k) along with captions.
   - Tokenize and preprocess captions for training.

3. **Model Training**:
   - Run the training script to train the image captioning model.
   - Adjust hyperparameters, model architecture, and training settings as needed.

4. **Evaluation**:
   - Evaluate the trained model on a subset of test data to compute BLEU scores.
   - Generate captions for sample images and visualize the results.

## References

1. [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/pdf/1411.4555.pdf) - Original paper on image captioning using CNN and LSTM.
2. [NLTK Documentation](https://www.nltk.org/) - Documentation for NLTK library used for text processing.
3. [Keras Documentation](https://keras.io/) - Documentation for Keras deep learning library.
4. [TensorFlow Documentation](https://www.tensorflow.org/) - Official TensorFlow documentation.
