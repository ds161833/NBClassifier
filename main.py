# todo remember what log is all about
# todo apply smoothing ta hell
from util import generate_files

training_file_name = 'covid_training_clean.tsv'
test_file_name = 'covid_test_public.tsv'

generate_files(training_file_name, test_file_name, is_filtered=False)
generate_files(training_file_name, test_file_name, is_filtered=True)
