# todo remember what log is all about
# todo apply smoothing ta hell
from nb_classifier import get_fit_classifier_from_file

training_file_name = 'covid_training.tsv'
test_file_name = 'covid_test_public.tsv'

classifier = get_fit_classifier_from_file(training_file_name, is_filtered=True)
result = classifier.predict("Everyone can help prevent the spread of #COVID19. Call your doctor if you develop symptoms, have been in close contact with a person known to have COVID-19, or have recently traveled from an area with widespread or ongoing community spread of COVID-19. https://t.co/ehL8kmRHaN. https://t.co/KwrKO7VNub")
