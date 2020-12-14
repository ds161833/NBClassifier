from util import generate_output_files

training_file_name = 'covid_training_clean.tsv'
test_file_name = 'covid_test_public.tsv'

generate_output_files(training_file_name, test_file_name, is_filtered=False)
generate_output_files(training_file_name, test_file_name, is_filtered=True)
