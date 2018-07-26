# Perceptron-Classifier-Hotel-Reviews
Vanilla and Average Perceptrons for Hotel Reviews Classification

In this assignment we will write perceptron classifiers (vanilla and averaged) to identify hotel reviews as either true or fake, and either positive or negative. 
You may using the word tokens as features, or any other features you can devise from the text. 

##Data
The data files and their decriptions are as follows:

* One file `train-labeled.txt` containing labeled training data with a single training instance (hotel review) per line (total 960 lines). 
The first 3 tokens in each line are:
  * a unique 7-character alphanumeric identifier
  * a label True or Fake
  * a label Pos or Neg

These are followed by the text of the review.
* One file `dev-text.txt` with unlabeled development data, containing just the unique identifier followed by the text of the review (total 320 lines).
* One file `dev-key.txt` with the corresponding labels for the development data, to serve as an answer key.

##Programs
The perceptron algorithms appear in `Hal Daumé III, A Course in Machine Learning (v. 0.99 draft), Chapter 4: The Perceptron.`

We have two programs: `perceplearn.py` will learn perceptron models (vanilla and averaged) from the training data, and `percepclassify.py` will use the models to classify new data. The learning program will be invoked in the following way:

```
> python perceplearn.py /path/to/input
```

The argument is a single file containing the training data; the program will learn perceptron models, and write the model parameters to two files: `vanillamodel.txt` for the vanilla perceptron, and `averagedmodel.txt` for the averaged perceptron. 
The format of the model files follow the following guidelines:

* The model files would contain sufficient information for percepclassify.py to successfully label new data.
* The model files are human-readable, so that model parameters can be easily understood by visual inspection of the file.
The classification program will be invoked in the following way:

```
> python percepclassify.py /path/to/model /path/to/input
```

The first argument is the path to the model file (`vanillamodel.txt` or `averagedmodel.txt`), 
and the second argument is the path to a file containing the test data file; 
the program will read the parameters of a perceptron model from the model file, 
classify each entry in the test data, and write the results to a text file called `percepoutput.txt`.

##Notes
* **Problem formulation**. Since a perceptron is a binary classifier, we need to treat the problem as two separate binary classification problems (true/fake and positive/negative); each of the model files (vanilla and averaged) needs to have the model parameters for both classifiers.
* **Features and tokenization**. For tokenization, we resort to removing punctuations and also lowercasing all the letters. Also, we remove all the stop words from a predefined list. Since this was a course assignment, we could not use something like NLTK Tokenizer.
* **Overfitting**. The perceptron has a tendency to overfit the training data. Make sure to plot the training and validation losses and see if the model is overfitting. 




