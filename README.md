https://github.com/ds161833/NBClassifier

NBClassifier
to run the code, you're gonna have to put you training data into covid_training_clean.tsv, make sure to follow the pattern that is in the existing file already, e.g. do not include any labels/columns names
put your test data into the covid_test_pulbic.tsv, again by following the format of the existing, given file (or use the preset data)

then create a new conda environment with conda :
```$ conda create -n my_env python=3.8```
then activate then environment
```$ conda activate my_env```
install needed packages:
```$ conda install scikit-learn``` (used to calculate precision, recall, accuracy, and f1 score)

```$ conda install numpy``` (used for array and matrix manipulation)

then run the main script, which will generate the trace and eval files you can see in the output folder:
```$ python main.py```

check the results in the output folder.

The project was completed by the students of Concordia University Dmytro Semenov and Gleb Galkin, team Black Dragon