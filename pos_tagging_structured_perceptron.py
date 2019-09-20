import numpy as np

from copy import deepcopy

from athnlp.readers.brown_pos_corpus import BrownPosTag

reader = BrownPosTag()


class StructuredPerceptron(object):
    def __init__(self, number_of_words, number_of_labels, epochs=3,
                 beam_size=3):
        self.number_of_words = number_of_words
        self.number_of_labels = number_of_labels
        self.w = np.zeros((number_of_words, number_of_labels))
        self.epochs = epochs
        self.beam_size = beam_size

    def predict(self, sentence):
        partial_hypotheses = [([], 0.0)]
        for i in range(len(sentence)):
            scores = np.dot(self.one_hot(sentence.x[i]), self.w)
            current_hypotheses = []
            for partial_hypothesis in partial_hypotheses:
                for j in range(self.number_of_labels):
                    new_partial_hypothesis = deepcopy(partial_hypothesis)
                    new_partial_hypothesis[0].append(j)
                    current_hypothesis = new_partial_hypothesis[0]
                    current_score = new_partial_hypothesis[1] + scores[j]
                    new_partial_hypothesis = (
                        current_hypothesis, current_score)
                    current_hypotheses.append(new_partial_hypothesis)
                partial_hypotheses = sorted(current_hypotheses,
                                            key=lambda x: x[1],
                                            reverse=True)[:self.beam_size]
        ret = sorted(partial_hypotheses, key=lambda x: x[1], reverse=True)[0]
        return ret

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
                labels, combined_score = self.predict(sentence)
                for yhat, y, x in zip(labels, sentence.y, sentence.x):
                    if y != yhat:
                        self.errors += 1
                        self.w[:, y] += self.one_hot(x)
                        self.w[:, yhat] -= self.one_hot(x)
                self.total += len(sentence.x)
            print('Finished epoch %d' % epoch)
            print('Errors: %d\tTotal: %d\tTraining accuracy: %.4f' % (
                self.errors, self.total, 1-self.errors/self.total))

    def evaluate(self, dev_set):
        print("---\nEvaluating model on %d sentences" % len(dev_set))
        self.errors = 0
        self.total = 0
        for sentence in dev_set:
            labels, combined_score = self.predict(sentence)
            for yhat, y, x in zip(labels, sentence.y, sentence.x):
                if y != yhat:
                    self.errors += 1
            self.total += len(sentence.x)
        print('Errors: %d\tTotal: %d\tAccuracy: %.4f' % (
            self.errors, self.total, 1-self.errors/self.total))


if __name__ == "__main__":
    p = StructuredPerceptron(
        len(reader.dictionary.x_dict), len(reader.dictionary.y_dict))
    p.train(reader.train)
    p.evaluate(reader.dev)
