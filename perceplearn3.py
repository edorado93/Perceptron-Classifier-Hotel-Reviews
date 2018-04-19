from random import shuffle
import numpy
import time

class Data:
    def __init__(self, filename, mini_batch_size=1):
        self.filename = filename
        self.reviews = {}
        self.mini_batch_size = mini_batch_size
        self.unique_words = None
        self.feature_vectors = []
        self.one_hot = {}
        self.weight_vector = None
        self.bias = 0.0
        self.read_corpus()

    def get_one_hot_vectors(self, word):
        zeros = numpy.zeros(len(self.unique_words)).reshape((1, -1))
        zeros[0][self.unique_words.index(word)] = 1
        return zeros

    def sentence_vector(self, review):
        rev = [self.one_hot[word] for word in review]
        concat = numpy.sum(rev, axis=0)
        return concat

    def read_corpus(self):
        start = time.time()
        self.unique_words = set()
        with open(self.filename) as f:
            for line in f:
                identifier, true_or_fake, pos_or_neg, *review = line.strip().split()
                self.reviews[identifier] = (true_or_fake, pos_or_neg, review)
                for word in review:
                    self.unique_words.add(word)
        self.unique_words = list(self.unique_words)

        """ Get one hot encoding for word vectors """
        for word in self.unique_words:
            self.one_hot[word] = self.get_one_hot_vectors(word)

        """ Form sentence vectors and store them """
        for identifier in self.reviews:
            true_or_fake, pos_or_neg, review = self.reviews[identifier]

            """ Converting expected outputs to binary values """
            true_or_fake = 1 if true_or_fake == "True" else 0
            pos_or_neg = 1 if pos_or_neg == "Pos" else 0
            self.feature_vectors.append((true_or_fake, pos_or_neg, self.sentence_vector(review)))

        """ Initialize weight vector """
        self.weight_vector = [numpy.random.rand(len(self.unique_words), 1), numpy.random.rand(len(self.unique_words), 1)]

        """ Initialize bias """
        self.bias = [numpy.random.random_sample(), numpy.random.random_sample()]

        print("Time taken to load the data is: ", (time.time() - start) * 1000)

    """ Shuffle up the reviews to be considered in every epoch. """
    def shuffle(self):
        shuffle(self.feature_vectors)
        return self.feature_vectors

    """ In case we hav to implement mini-batching. """
    def get_next_batch(self, index):
        return self.reviews[index : min(index + self.mini_batch_size, len(self.reviews))], min(index + self.mini_batch_size, len(self.reviews))

class VanillaPerceptron:
    def __init__(self, data, epochs=5):
        self.data = data
        self.epochs = epochs

    def train(self):
        for epoch in range(1, self.epochs):
            success = [0, 0]
            for tup in self.data.feature_vectors:
                true_or_fake, pos_or_neg, vector = tup
                """ We are training two classifiers here """
                activation1 = numpy.matmul(vector, self.data.weight_vector[0]) + self.data.bias[0]
                activation2 = numpy.matmul(vector, self.data.weight_vector[1]) + self.data.bias[1]

                """ Update weights for classifier 1 """
                if true_or_fake * activation1 <= 0:
                    self.data.weight_vector[0] += true_or_fake * vector
                    self.data.bias[0] += true_or_fake
                else:
                    success[0] += 1

                """ Update weights for classifier 2 """
                if pos_or_neg * activation2 <= 0:
                    self.data.weight_vector[1] += pos_or_neg * vector
                    self.data.bias[1] += pos_or_neg
                else:
                    success[1] += 1

            print("Epoch:", epoch," Accuracy:", success[0] / len(self.data.feature_vectors), "% on True / Fake dataset and ",success[1] / len(self.data.feature_vectors), "% on Pos / Neg dataset")

class AveragePerceptron:
    def __init__(self):
        pass

if __name__ == "__main__":
    data = Data("coding-2-data-corpus/train-labeled.txt")