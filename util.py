import csv
import numpy as np

from nb_classifier import get_fit_classifier_from_file


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


def get_trace_line(classifier, row: np.ndarray):
    result, enthropy = classifier.predict(row[1])
    match = result == row[2]
    return f'{row[0]}  {result}  {enthropy}  {row[2]}  {"correct" if match else "wrong"}', match


def test_classifier(classifier, test_file_name):
    test_set = get_dataset_set_from_file(test_file_name)
    match_count = 0
    for row in test_set:
        trace_line, is_match = get_trace_line(classifier, row)
        print(trace_line)
        if is_match:
            match_count += 1
    print(f'got right {match_count}/{len(test_set)}')


def generate_trace_file(training_file_name, test_file_name, is_filtered):
    classifier = get_fit_classifier_from_file(training_file_name, is_filtered)

    test_set = get_dataset_set_from_file(test_file_name)
    match_count = 0
    for row in test_set:
        trace_line, is_match = get_trace_line(classifier, row)
        print(trace_line)
        if is_match:
            match_count += 1
    print(f'got right {match_count}/{len(test_set)}')
