# this code has to be placed at the end of the other python code to do what B4.b requires

model_2 = keras.Sequential([
        # hidden layer
        keras.layers.Dense(units=397, activation='relu', input_shape=[28 * 28, ]),
        # hidden layer 2
        keras.layers.Dense(units=397, activation='relu'),
        # output layer
        keras.layers.Dense(units=10, activation='softmax')
])
keras.optimizers.SGD(lr=0.05, momentum=0.6, nesterov=False)
model_2.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Read dataset
dataset_2 = np.loadtxt("mnist_train.csv", delimiter=",", skiprows=1)
# Split into input and output
output_2 = dataset_2[:, 0]
# onehot encode the output to match the 10 output neurons
Y_2 = OneHotEncoder(sparse=False).fit_transform(X=output_2.reshape(len(output_2), 1))
# Remove the output from the input dataset
dataset_2 = dataset_2[:, 1:]
print("Successfully read the train dataset")
# normalize the dataset
X_2 = PowerTransformer().fit_transform(X=dataset_2)
re_train_dataset = X_2 * best
history_2 = model_2.fit(re_train_dataset, Y_2, epochs=100, batch_size=512, verbose=0)
scores_2 = model_2.evaluate(X, Y, verbose=1)
print(scores_2)
plt.figure(1)
plt.plot(history_2.history['loss'])
plt.ylabel("Cross entropy Loss")
plt.xlabel("Epoch")
plt.figure(2)
plt.plot(history_2.history['accuracy'])
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.show()
