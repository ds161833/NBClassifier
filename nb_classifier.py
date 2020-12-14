import numpy as np


class MNBClassifier:
    smooth_factor = 0.00001

    def __init__(self):
        self.label_probability = {"yes": 0, "no": 0}
        self.words_probability_given_class = {"yes": dict(), "no": dict()}

    def fit(self, sentences, labels, is_filtered):
        self._build_labels_likelyhood(labels)
        self._build_vocabulary(sentences, labels, is_filtered)

    def _build_vocabulary(self, sentences, labels, is_filtered):
        word_count_per_label = {"yes": 0, "no": 0}

        for i in range(len(sentences)):
            current_sentence_split = str.lower(sentences[i]).split()
            current_label = labels[i]
            word_count_per_label[current_label] = word_count_per_label[current_label] + len(current_sentence_split)

            for word in current_sentence_split:
                self.words_probability_given_class[current_label][word] = self.words_probability_given_class[current_label].get(word, 0) + 1

        if is_filtered:
            self._filter()

        for label_tuple in word_count_per_label.items():
            label = label_tuple[0]
            label_count = label_tuple[1]

            for entry in self.words_probability_given_class[label].items():
                word = entry[0]
                word_count = entry[1]
                self.words_probability_given_class[label][word] = word_count / label_count

    def predict(self, sentence):
        p_of_yes = self._get_score(sentence, "yes")
        p_of_no = self._get_score(sentence, "no")
        if p_of_yes > p_of_no:
            return "yes", p_of_yes
        else:
            return "no", p_of_no

    def _get_score(self, sentence, label):
        estimate = np.log10(self.label_probability[label])

        for word in str.lower(sentence).split():
            probability_given_label = self._get_p_of_word_given_label(word, label)
            next_probability = np.log10(MNBClassifier.smooth_factor) if probability_given_label == 0 else np.log10(probability_given_label)
            estimate = estimate + next_probability

        return estimate

    def _get_p_of_word_given_label(self, word, label):
        return self.words_probability_given_class[label].get(word, 0)

    def _get_probability(self, labels, label):
        counted_labels = dict(zip(*np.unique(labels, return_counts=True)))
        return counted_labels[label]/len(labels)

    def _build_labels_likelyhood(self, labels):
        self.label_probability["yes"] = self._get_probability(labels, "yes")
        self.label_probability["no"] = self._get_probability(labels, "no")

    def _filter(self):
        yes_dict = self.words_probability_given_class["yes"].copy()
        for word_tuple in yes_dict.items():
            word = word_tuple[0]
            word_count = word_tuple[1]
            if word_count == 1:
                opposite_count = self.words_probability_given_class["no"].get(word, 0)
                if opposite_count == 0:
                    self.words_probability_given_class["yes"].pop(word)

        no_dict = self.words_probability_given_class["no"].copy()
        for word_tuple in no_dict.items():
            word = word_tuple[0]
            word_count = word_tuple[1]
            if word_count == 1:
                opposite_count = self.words_probability_given_class["yes"].get(word, 0)
                if opposite_count == 0:
                    self.words_probability_given_class["no"].pop(word)
