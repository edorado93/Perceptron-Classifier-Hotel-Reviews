import ast
import sys
import util3
from collections import Counter

class PerceptronClassify:
    def __init__(self, model_filename, test_data, output_file):
        self.model_filename = model_filename
        self.weight_vector = None
        self.bias = None
        self.test_file = test_data
        self.output_file = output_file
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

    def load(self):
        with open(self.model_filename) as f:
            self.training_unique_words = ast.literal_eval(f.readline())
            self.weight_vector = [ast.literal_eval(f.readline()), ast.literal_eval(f.readline())]
            self.bias = [float(f.readline()), float(f.readline())]

    def classify_review(self, review):
        counts = Counter(review)
        activation1 = 0
        activation2 = 0
        for word in counts:
            if word in self.training_unique_words:
                activation1 += self.weight_vector[0][word] * counts[word]
                activation2 += self.weight_vector[1][word] * counts[word]

        return "True" if (activation1 + self.bias[0]) > 0 else "Fake", "Pos" if (activation2 + self.bias[1]) > 0 else "Neg"

if __name__ == "__main__":
    perceptron = PerceptronClassify(sys.argv[1], sys.argv[2], "percepoutput.txt")
    perceptron.load()
    perceptron.classify()