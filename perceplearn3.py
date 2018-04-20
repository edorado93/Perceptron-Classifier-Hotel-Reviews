from random import shuffle
import numpy
import time
import sys

class Data:

    UNK = "unk"

    def __init__(self, filename, mini_batch_size=1):
        self.filename = filename
        self.reviews = {}
        self.mini_batch_size = mini_batch_size
        self.unique_words = None
        self.feature_vectors = []
        self.one_hot = {}
        self.read_corpus()

    def one_hot_vectors(self):
        for index, word in enumerate(self.unique_words):
            zeros = numpy.zeros((1, len(self.unique_words)))
            zeros.put(index, 1.0)
            self.one_hot[word] = zeros

    def form_sentence_vectors(self):
        for identifier in self.reviews:
            true_or_fake, pos_or_neg, review = self.reviews[identifier]
            sent_vec = numpy.copy(self.one_hot[review[0]])
            for word in review[1:]:
                sent_vec += self.one_hot[word]

            """ Converting expected outputs to binary values """
            true_or_fake = 1 if true_or_fake == "True" else -1
            pos_or_neg = 1 if pos_or_neg == "Pos" else -1

            self.feature_vectors.append((true_or_fake, pos_or_neg, sent_vec))

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
        self.unique_words.append(Data.UNK)

        """ Get one hot encoding for word vectors """
        self.one_hot_vectors()

        """ Form sentence vectors --> feature_vectors """
        self.form_sentence_vectors()

    """ Shuffle up the reviews to be considered in every epoch. """
    def shuffle(self):
        shuffle(self.feature_vectors)

    """ In case we hav to implement mini-batching. """
    def get_next_batch(self, index):
        return self.reviews[index : min(index + self.mini_batch_size, len(self.reviews))], min(index + self.mini_batch_size, len(self.reviews))

class Perceptron:
    def __init__(self, data, model_filename, epochs=5, is_average=False):
        self.data = data
        self.epochs = epochs
        """ Initialize weight vector """
        self.weight_vector = [numpy.random.rand(len(data.unique_words), 1),
                              numpy.random.rand(len(data.unique_words), 1)]
        self.cached_weight_vector = [numpy.random.rand(len(data.unique_words), 1),
                                     numpy.random.rand(len(data.unique_words), 1)]

        """ Initialize bias """
        self.bias = [numpy.random.random_sample(), numpy.random.random_sample()]
        self.cached_bias = [numpy.random.random_sample(), numpy.random.random_sample()]

        self.is_average_perceptron = is_average
        self.model_filename = model_filename

    def train(self):
        for epoch in range(1, self.epochs):
            success = [0, 0]
            self.data.shuffle()
            for tup in self.data.feature_vectors:
                true_or_fake, pos_or_neg, vector = tup
                """ We are training two classifiers here """
                activation1 = numpy.matmul(vector, self.weight_vector[0]) + self.bias[0]
                activation2 = numpy.matmul(vector, self.weight_vector[1]) + self.bias[1]

                """ Update weights for classifier 1 """
                if true_or_fake * activation1 <= 0:
                    self.weight_vector[0] += numpy.transpose(numpy.multiply(true_or_fake, vector))
                    self.bias[0] += true_or_fake

                    if self.is_average_perceptron:
                        self.cached_weight_vector[0] += numpy.transpose(numpy.multiply(numpy.multiply(true_or_fake, vector), epoch))
                        self.bias[0] += true_or_fake * epoch

                else:
                    success[0] += 1

                """ Update weights for classifier 2 """
                if pos_or_neg * activation2 <= 0:
                    self.weight_vector[1] += numpy.transpose(numpy.multiply(pos_or_neg, vector))
                    self.bias[1] += pos_or_neg

                    if self.is_average_perceptron:
                        self.cached_weight_vector[1] += numpy.transpose(numpy.multiply(numpy.multiply(pos_or_neg, vector), epoch))
                        self.bias[1] += pos_or_neg * epoch

                else:
                    success[1] += 1

            # print("Epoch:", epoch," Accuracy:", success[0] / len(self.data.feature_vectors), "% on True / Fake dataset and ",success[1] / len(self.data.feature_vectors), "% on Pos / Neg dataset")

        if self.is_average_perceptron:
            self.weight_vector[0] = numpy.subtract(self.weight_vector[0], numpy.multiply(1 / (self.epochs+1), self.cached_weight_vector[0]))
            self.weight_vector[1] = numpy.subtract(self.weight_vector[1], numpy.multiply(1 / (self.epochs+1), self.cached_weight_vector[1]))
            self.bias[0] -= (1 / (self.epochs+1)) * self.cached_bias[0]
            self.bias[1] -= (1 / (self.epochs + 1)) * self.cached_bias[1]

        self.save()

    def save(self):
        with open(self.model_filename, "w") as f:
            f.write(str(data.unique_words));f.write("\n")

            w = []
            for x in numpy.nditer(self.weight_vector[0]):
                w.append(str(x))
            f.write(str(w));f.write("\n")

            w = []
            for x in numpy.nditer(self.weight_vector[1]):
                w.append(str(x))
            f.write(str(w));f.write("\n")
            f.write(str(self.bias[0]));f.write("\n")
            f.write(str(self.bias[1]));

if __name__ == "__main__":
    data = Data(sys.argv[1])
    Perceptron(data, "vanillamodel.txt", epochs=20).train()
    Perceptron(data, "averagedmodel.txt", epochs=20, is_average=True).train()