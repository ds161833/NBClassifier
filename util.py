import csv
import numpy as np

from nb_classifier import MNBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def get_fit_classifier_from_file(file_name: str, is_filtered: bool) -> MNBClassifier:
    training_set = get_dataset_set_from_file(file_name)
    sentences = get_sentences(training_set)
    labels = get_labels(training_set)

    classifier = MNBClassifier()
    classifier.fit(sentences, labels, is_filtered)

    return classifier


def get_dataset_set_from_file(file_name: str) -> np.ndarray:
    tsv = csv.reader(open(file_name, encoding='UTF8'), delimiter="\t")
    x = list(tsv)
    result = np.array(x).astype("str")
    result = result[:, 0:3]
    return result


def get_sentences(training_set: np.ndarray) -> np.ndarray:
    return training_set[:, 1]


def get_labels(training_set: np.ndarray) -> np.ndarray:
    return training_set[:, 2]


def get_trace_line(result, enthropy, row: np.ndarray):
    match = result == row[2]
    return f'{row[0]}  {result}  {enthropy}  {row[2]}  {"correct" if match else "wrong"}'


def generate_files(training_file_name, test_file_name, is_filtered):
    classifier = get_fit_classifier_from_file(training_file_name, is_filtered)

    test_set = get_dataset_set_from_file(test_file_name)

    classifier_name = 'FV' if is_filtered else 'OV'
    file = open(f"output/trace_NB-BOW-{classifier_name}.txt", "w")

    labels_predicted = list()
    labels_true = list()
    for row in test_set:
        result, enthropy = classifier.predict(row[1])
        file.write(get_trace_line(result, enthropy, row))
        file.write('\n')
        labels_predicted.append(0 if result == "yes" else 1)
        labels_true.append(0 if row[2] == "yes" else 1)
    file.close()

    file = open(f"output/eval_NB-BOW-{classifier_name}.txt", "w")
    statistics = Eval(labels_true, labels_predicted)

    file.write(str(statistics.accuracy))
    file.write(f'\n{statistics.precision["yes"]} {statistics.precision["no"]}')
    file.write(f'\n{statistics.recall["yes"]} {statistics.recall["no"]}')
    file.write(f'\n{statistics.f1_measure["yes"]} {statistics.f1_measure["no"]}')

    file.close()


class Eval:
    def __init__(self, labels_true, labels_predicted):
        self.labels_true = np.array(labels_true)
        self.labels_predicted = np.array(labels_predicted)
        self.accuracy = 0
        self.precision = {"yes": 0, "no": 0}
        self.recall = {"yes": 0, "no": 0}
        self.f1_measure = {"yes": 0, "no": 0}
        self.generate_statistics()

    def generate_statistics(self):
        self._set_accuracy()
        self._set_precision()
        self._set_recall()
        self._set_f1_measure()

    def _set_accuracy(self):
        self.accuracy = accuracy_score(self.labels_true, self.labels_predicted)

    def _set_precision(self):
        score = precision_score(self.labels_true, self.labels_predicted, average=None)
        self.precision = {"yes": score[0], "no": score[1]}

    def _set_recall(self):
        score = recall_score(self.labels_true, self.labels_predicted, average=None)
        self.recall = {"yes": score[0], "no": score[1]}

    def _set_f1_measure(self):
        score = f1_score(self.labels_true, self.labels_predicted, average=None)
        self.f1_measure = {"yes": score[0], "no": score[1]}


