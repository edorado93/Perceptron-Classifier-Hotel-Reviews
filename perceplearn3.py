from random import shuffle
import sys
import util3
from collections import Counter
import random

class Data:
    def __init__(self, filename):
        self.filename = filename
        self.unique_words = None
        self.feature_vectors_true_fake = []
        self.feature_vectors_pos_neg = []
        self.read_corpus()

    def read_corpus(self):
        self.unique_words = set()
        with open(self.filename) as f:
            for line in f:
                line = util3.remove_stop_words(util3.remove_punctuation(line))
                identifier, true_or_fake, pos_or_neg, *review = line.strip().split()
                review = list(map(str.lower, review))
                """ Converting expected outputs to binary values """
                true_or_fake = 1 if true_or_fake == "True" else -1
                pos_or_neg = 1 if pos_or_neg == "Pos" else -1
                self.feature_vectors_true_fake.append((true_or_fake, review))
                self.feature_vectors_pos_neg.append((pos_or_neg, review))

                for word in review:
                    self.unique_words.add(word)
    """ Shuffle up the reviews to be considered in every epoch. """
    def shuffle(self):
        shuffle(self.feature_vectors_pos_neg)
        shuffle(self.feature_vectors_true_fake)

class Perceptron:
    def __init__(self, data, model_filename, epochs=5, is_average=False):
        self.data = data
        self.epochs = epochs
        """ Initialize weight vector """
        self.weight_vector = [{}, {}]
        self.cached_weight_vector = [{}, {}]

        """ Initialize bias """
        self.bias = [0, 0]
        self.cached_bias = [0, 0]

        self.is_average_perceptron = is_average
        self.model_filename = model_filename

    def initialise_weights(self):
        for word in data.unique_words:
            self.weight_vector[0][word] = 0
            self.weight_vector[1][word] = 0
            self.cached_weight_vector[0][word] = 0
            self.cached_weight_vector[1][word] = 0

    def classify_and_update(self, review, weight, bias, cached_weight, cached_bias, output, epoch, is_average):
        activation = 0
        counts = Counter(review)
        for word in counts:
            activation += weight[word] * counts[word]

        """ Update weights for classifier 1 """
        if output * (activation + bias) <= 0:
            for word in counts:
                weight[word] += output * counts[word]
                if is_average:
                    cached_weight[word] += output * counts[word] * epoch
            bias += output
            if is_average:
                cached_bias += output * epoch
            return False
        else:
            return True

    def train(self):
        stopping_epoch = self.epochs
        self.initialise_weights()
        # self.data.shuffle()
        index = 1
        for epoch in range(1, self.epochs):
            success = [0, 0]
            for tup1, tup2 in zip(self.data.feature_vectors_pos_neg, self.data.feature_vectors_true_fake):
                true_or_fake, review_tf = tup2
                pos_or_neg, review_pn = tup1

                """ review, weight, bias, cached_weight, cached_bias, output, epoch, is_average """
                success[0] += True == self.classify_and_update(review_tf, self.weight_vector[0], self.bias[0], self.cached_weight_vector[0]
                                         , self.cached_bias[0], true_or_fake, index, self.is_average_perceptron)
                success[1] += True == self.classify_and_update(review_pn, self.weight_vector[1], self.bias[1], self.cached_weight_vector[1]
                                         , self.cached_bias[1], pos_or_neg, index, self.is_average_perceptron)
                index += 1


            """ Classified completely """
            if success[0] == len(self.data.feature_vectors_true_fake) and success[0] == success[1]:
                stopping_epoch = index
                break
            # print("Epoch:", epoch," Accuracy:", success[0] / len(self.data.feature_vectors_true_fake), "% on True / Fake dataset and ",success[1] / len(self.data.feature_vectors_true_fake), "% on Pos / Neg dataset")

        if self.is_average_perceptron:
            for word in self.data.unique_words:
                self.weight_vector[0][word] -= (1 / (stopping_epoch + 1)) * self.cached_weight_vector[0][word]
                self.weight_vector[1][word] -= (1 / (stopping_epoch + 1)) * self.cached_weight_vector[1][word]
            self.bias[0] -= (1 / (stopping_epoch + 1)) * self.cached_bias[0]
            self.bias[1] -= (1 / (stopping_epoch + 1)) * self.cached_bias[1]

        self.save()

    def save(self):
        with open(self.model_filename, "w") as f:
            f.write(str(data.unique_words));f.write("\n")
            f.write(str(self.weight_vector[0]));f.write("\n")
            f.write(str(self.weight_vector[1]));f.write("\n")
            f.write(str(self.bias[0]));f.write("\n")
            f.write(str(self.bias[1]))

if __name__ == "__main__":
    data = Data(sys.argv[1])
    Perceptron(data, "vanillamodel.txt", epochs=100).train()
    Perceptron(data, "averagedmodel.txt", epochs=100, is_average=True).train()