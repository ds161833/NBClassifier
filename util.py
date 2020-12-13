import csv
import numpy as np

from nb_classifier import MNBClassifier


def get_training_set_from_file(file_name: str) -> np.ndarray:
    tsv = csv.reader(open(file_name, encoding='UTF8'), delimiter="\t")
    x = list(tsv)
    result = np.array(x).astype("str")
    result = result[:, 0:3]
    return result


def get_sentences(training_set: np.ndarray) -> np.ndarray:
    return training_set[:, 1]


def get_labels(training_set: np.ndarray) -> np.ndarray:
    return training_set[:, 2]

def get_trace_line(classifier: MNBClassifier, row: np.ndarray):
    result, enthropy = classifier.predict(row[1])
    print(f'{row[0]}  {result}  {enthropy}  {row[2]}')

