import numpy as np

from athnlp.readers.brown_pos_corpus import BrownPosTag

reader = BrownPosTag()


class Perceptron(object):
    def __init__(self, number_of_words, number_of_labels, epochs=3):
        self.number_of_words = number_of_words
        self.w = np.zeros((number_of_words, number_of_labels))
        self.epochs = epochs

    def predict(self, x):
        one_hot_x = self.one_hot(x)
        return np.dot(one_hot_x, self.w)

    def one_hot(self, x):
        one_hot_x = np.zeros(self.number_of_words)
        one_hot_x[x] = 1
        return one_hot_x

    def train(self, training_set, seed=1):
        print('Training model on %d sentences for %d epochs' % (
            len(training_set), self.epochs))
        for epoch in range(1, self.epochs+1):
            np.random.seed(seed)
            seed += 1
            training_shuffle = np.random.permutation(training_set)
            self.errors = 0
            self.total = 0
            # Iterate over example sentences
            for sentence in training_shuffle:
                # Iterate over words
                for i in range(len(sentence)):
                    x_i = sentence.x[i]
                    scores = self.predict(x_i)
                    yhat_i = np.argmax(scores, axis=0)
                    y_i = sentence.y[i]
                    if yhat_i != y_i:
                        one_hot_x_i = self.one_hot(x_i)
                        self.w[:, y_i] += one_hot_x_i
                        self.w[:, yhat_i] -= one_hot_x_i
                        self.errors += 1
                    self.total += 1
            print('Finished epoch %d' % epoch)
            print('Errors: %d\tTotal: %d\tTraining accuracy: %.4f' % (
                self.errors, self.total, 1-self.errors/self.total))

    def evaluate(self, dev_set):
        print("---\nEvaluating model on %d sentences" % len(dev_set))
        self.errors = 0
        self.total = 0
        for sentence in dev_set:
            # Iterate over words
            for i in range(len(sentence)):
                x_i = sentence.x[i]
                scores = self.predict(x_i)
                yhat_i = np.argmax(scores, axis=0)
                y_i = sentence.y[i]
                if yhat_i != y_i:
                    self.errors += 1
                self.total += 1
        print('Errors: %d\tTotal: %d\tAccuracy: %.4f' % (
            self.errors, self.total, 1-self.errors/self.total))


if __name__ == "__main__":
    p = Perceptron(len(reader.dictionary.x_dict),
                   len(reader.dictionary.y_dict))
    p.train(reader.train)
    p.evaluate(reader.dev)
