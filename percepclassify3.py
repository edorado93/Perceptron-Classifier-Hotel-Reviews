import numpy
import ast
import sys
import util3

class PerceptronClassify:

    UNK = "unk"

    def __init__(self, model_filename, test_data, output_file):
        self.model_filename = model_filename
        self.weight_vector = None
        self.bias = None
        self.test_file = test_data
        self.output_file = output_file
        self.one_hot_vectors = {}
        self.training_unique_words = None

    def classify(self):
        with open(self.output_file, "w") as o:
            with open(self.test_file) as f:
                for line in f:
                    line = util3.remove_stop_words(util3.remove_punctuation(line))
                    identifier, *review = line.strip().split()
                    review = list(map(str.lower, review))
                    true_or_fake, pos_or_neg = self.classify_review(review)
                    o.write(identifier + " " + true_or_fake + " " + pos_or_neg + "\n")

    def create_one_hot_vectors(self):
        for index, word in enumerate(self.training_unique_words):
            zeros = numpy.zeros((1, len(self.training_unique_words)))
            zeros.put(index, 1.0)
            self.one_hot_vectors[word] = zeros

    def load(self):
        with open(self.model_filename) as f:
            self.training_unique_words = ast.literal_eval(f.readline())
            self.weight_vector = [numpy.asarray(ast.literal_eval(f.readline()), dtype='float64').reshape((-1, 1)),
                                  numpy.asarray(ast.literal_eval(f.readline()), dtype='float64').reshape((-1, 1))]
            self.bias = [float(f.readline()), float(f.readline())]
            self.create_one_hot_vectors()

    def classify_review(self, review):

        sent_vec = None
        start_index = None
        for index, word in enumerate(review):
            if word in self.one_hot_vectors:
                sent_vec = numpy.copy(self.one_hot_vectors[word])
                start_index = index
                break

        if start_index is None:
            return "True", "Pos"

        for word in review[start_index:]:
            if word in self.one_hot_vectors:
                sent_vec += self.one_hot_vectors[word]

        activation1 = numpy.matmul(sent_vec, self.weight_vector[0]) + self.bias[0]
        activation2 = numpy.matmul(sent_vec, self.weight_vector[1]) + self.bias[1]

        return "True" if activation1 > 0 else "Fake", "Pos" if activation2 > 0 else "Neg"

if __name__ == "__main__":
    perceptron = PerceptronClassify(sys.argv[1], sys.argv[2], "percepoutput.txt")
    perceptron.load()
    perceptron.classify()