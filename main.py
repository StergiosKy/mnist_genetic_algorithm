import random
import numpy as np
from deap import creator, base, tools, algorithms
from sklearn.preprocessing import PowerTransformer
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot as plt
import pandas as pd


def evaluate(individual):
    """
    :param: individual: the individual of the population for evaluation
    :return: custom metric that includes a penalty
    """
    # create the individual's input
    individual_input = X * individual
    # evaluate
    scores = model.evaluate(individual_input, Y, verbose=0)
    # calculate the fit based on a custom metric that applies a penalty to the cross entropy and return it
    # penalty is positive because we try to minimize the value
    return scores[0] * (1 + (np.count_nonzero(individual[0]) / 784)),


# Parameters
# The parameters variable has to be a list of lists to work as intended
# [0][] -> the population parameters
# [1][] -> the crossover parameters
# [2][] -> the mutation parameters
# For a single set of parameters please use a format of [[], [], []]
# example: parameters to execute only 1 set of parameters [[20], [0.6], [0.0]]
# This allows to run many different configurations consecutively
'''
parameters = [[20, 20, 20, 20, 20, 200, 200, 200, 200, 200],
                [0.6, 0.6, 0.6, 0.9, 0.1, 0.6, 0.6, 0.6, 0.9, 0.1],
                [0.0, 0.01, 0.1, 0.01, 0.01, 0.0, 0.01, 0.1, 0.01, 0.01]]
'''
parameters = [[200, 200], [0.9, 0.1], [0.01, 0.01]]

# ---------------------------------!!!!!!!!!!!!!!!!--------------------------------
# numbers don't matter for the 3 following variables when parameters (see above) is used
# to not use it one must delete the outer for loop on the driver code
population = 20
crpb = 0.6
mtpb = 0.0
gene_length = 784

# generation counter
cur_gen = 0

# flag to decide if elitism will be used
# the size of the hof chosen is 5% of the population
use_hof = 0
# flag to decide if to mutate the hof, if hof is used
mutate_hof = 0
# exit flag for the while loop
exit_flag = 0

# executions, defines how many times to run each genetic algorithm
executions = 10

# to store the best fitness of an execution to create the plots later on
bests_list = []
# history of the number of generations it took to finish
gen_history = []
# history of the best individual for each execution
final_bests_history = []
# more statistics
ce_history = []
acc_history = []

model = keras.models.load_model('fully_trained_model')
# model.summary() will show the features of the loaded neural network
print("Successfully loaded the model")

# The following section is the same as in part one except we use the test dataset only, we will not use cv here
# nor will we train the NN again to avoid adding more time complexity to an already time consuming problem
# that goes especially for the 200 population case

# Read dataset
dataset = np.loadtxt("mnist_test.csv", delimiter=",", skiprows=1)
# Split into input and output
output = dataset[:, 0]
# onehot encode the output to match the 10 output neurons
Y = OneHotEncoder(sparse=False).fit_transform(X=output.reshape(len(output), 1))
# Remove the output from the input dataset
dataset = dataset[:, 1:]
print("Successfully read the dataset")

# normalize the dataset
X = PowerTransformer().fit_transform(X=dataset)

# Create a FitnessMin class with the purpose of minimizing the cross entropy loss
# which will be a metric that combines ce loss and the remaining input neurons as a form of penalty
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# Create the "individual" class, which will inherit the class list
# and contain our previously defined FitnessMax class in its fitness attribute
creator.create("Individual", list, fitness=creator.FitnessMin)

# Create a toolbox
toolbox = base.Toolbox()
# Attribute generator function for the individuals
toolbox.register("zero_ind", random.randint, 0, 1)
# Structure initializer
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.zero_ind, n=gene_length)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
# Operators
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxUniform, indpb=0.4)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.3)
toolbox.register("select", tools.selTournament, tournsize=6)
# Statistics
stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("min", np.min)

# Driver code below
# Run multiple cases consecutively and export plots, general info
for i in range(len(parameters[0])):
    # Parameter assignment
    population = parameters[0][i]
    crpb = parameters[1][i]
    mtpb = parameters[2][i]

    # Reset statistics
    ce_history = []
    acc_history = []
    final_bests_history = []
    bests_list = []
    gen_history = []

    for j in range(executions):
        print(f"execution {j}")
        # a single execution of the genetic algorithm

        # First generate and fit the initial population
        pop = toolbox.population(n=population)
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        fits = [ind.fitness.values[0] for ind in pop]

        # Statistics
        record = stats.compile(pop)

        # hof for elitism if it is used
        if use_hof:
            hof = tools.HallOfFame(int(population * 0.05))
            hof.update(pop)
            hof_size = len(hof.items) if hof.items else 0
        else:
            hof_size = 0

        best_history = []
        best_history.append(record['min'])
        print(f"gen {cur_gen} best: {best_history[-1]}")
        while True:
            # A new generation begins here
            cur_gen += 1
            # Select the individuals for the next generation
            offspring = toolbox.select(pop, len(pop) - hof_size)
            # Use the varAnd algorithm to apply crossover and mutation with crossover probability crpb and
            # mutation probability mtpb
            offspring = algorithms.varAnd(offspring, toolbox, crpb, mtpb)

            # Evaluate all the individuals who have an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            record = stats.compile(pop)

            # If elitism is used
            if use_hof:
                # Add the best back to population:
                if mutate_hof:
                    # Mutation case
                    for i in range(hof_size):
                        offspring.extend(toolbox.mutate(hof.items[i]))
                else:
                    # Non mutation case
                    offspring.extend(hof.items)

                # Update the hall of fame with the generated individuals
                hof.update(offspring)

            # New population using the offspring
            pop[:] = offspring

            # Save the statistics
            best_history.append(record['min'])
            print(f"gen {cur_gen} best: {best_history[-1]}")

            # Exit condition is to observe below 1% relative upgrade on 5 consecutive iterations
            # from general observation over many executions 5 consecutive iterations were a good cutoff point
            # to decide if the program found a local optima and got stuck there or it just happened to not get a better candidate
            # when using a population of 20
            if abs(best_history[-1] - best_history[-2]) < 0.01 * best_history[-1]:
                exit_flag += 1
                if exit_flag >= 5:
                    break
            else:
                exit_flag = 0

            # The 20 population run on average for 16-18 generations, 16 generations should suffice for testing
            # purposes on 200 individuals, even though it will still take a very long time compared to the 20 population runs
            # but we can still draw the conclusions we need to from the data
            if population == 200:
                if cur_gen == 16:
                    break

        # Find the best individual of the population and run the neural network with that individual to see the results
        best = pop[np.argmin([toolbox.evaluate(x) for x in pop])]
        best_input = X * best
        scores = model.evaluate(best_input, Y, verbose=1)
        original_scores = model.evaluate(X, Y, verbose=1)
        print(f"Number of zeroed inputs: {784 - np.count_nonzero(best)}")
        print(f"Population: {population}\nCrossover Probability: {crpb}\nMutation Probability: {mtpb}\nGenerations to convergence: {cur_gen}")
        ce_history.append(scores[0] - original_scores[0])
        acc_history.append(100 * (original_scores[1] - scores[1]))
        print(f"Total losses in performance:\nAccuracy: {acc_history[-1]}%")
        print(f"Cross entropy: {ce_history[-1]}")
        print("--------------------------------------------------------------------------------------------------------")

        bests_list.append(best_history)
        gen_history.append(cur_gen)
        final_bests_history.append(best_history[-1])
        cur_gen = 0
        exit_flag = 0

    # Plot
    df = pd.DataFrame(bests_list).mean(axis=0)
    plt.figure()
    df.plot(title=f"pop: {population}, crpd: {crpb}, mtpb: {mtpb}")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")

    # Export the statistics to a txt
    text_file = open(f"{population}_{crpb}_{mtpb}.txt", "w")
    text_file.write(f"Avg generations: {np.mean(gen_history)}\nAvg best Fitness: {np.mean(final_bests_history)}\n"
                    f"Average losses in NN performance:\nCE loss: {np.mean(ce_history)}\nAcc loss: {np.mean(acc_history)}%")
    text_file.close()

    # Statistics to the terminal
    print(f"Avg generations: {np.mean(gen_history)}")
    print(f"Avg best Fitness: {np.mean(final_bests_history)}")
    print("Average losses in NN performance:")
    print(f"CE loss: {np.mean(ce_history)}")
    print(f"Acc loss: {np.mean(acc_history)}%")

    # Export the plot as a png
    plt.savefig(f'{population}_{crpb}_{mtpb}.png')
