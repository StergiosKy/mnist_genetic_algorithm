# mnist_genetic_algorithm
A genetic algorithm implementation to detect useless inputs in a pre-trained NN

Supported features are regular crossover + mutation, crossover + mutation with elitism and crossover + mutation with elitism and mutation applied to it.
The code supports running multiple sets of parameters in succession without the need to restart and is very scalable/configurable.
Graphs are exported as pngs and some general information for the run as a txt file.

This version does not use CV nor does it retrain the NN, mainly due to the fact that the
NN being tested in this case is very huge and thus for huge population sizes the runs would take a prohibitive amount of time,
even on a GPU and also due to the fact that there wouldn't be many significant changes in the end results. Support for it is easy to add though.
