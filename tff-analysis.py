import tensorflow as tf
import tensorflow_federated as tff
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D

import datetime
start_time = datetime.datetime.now()
print("Program started at:", start_time)

(x_train, y_train), (x_test , y_test) = tf.keras.datasets.cifar10.load_data()

no_of_clients = 3
requesting_client = 0
no_of_epochs_client_side = 16
NUM_ROUNDS = 100

def create_tf_dataset_for_client():
    global requesting_client
    gap = len(x_train) / no_of_clients
    gap = int(gap)
    start = requesting_client * gap
    dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(x_train[start: start + gap]), tf.convert_to_tensor(y_train[start : start + gap])))
    requesting_client += 1
    return dataset.repeat(count=no_of_epochs_client_side).batch(32)


federated_data = []
for i in range(no_of_clients):
    federated_data.append(create_tf_dataset_for_client())


def create_keras_model():
	return tf.keras.models.Sequential(
			[
				Conv2D(filters=6, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(32 , 32 , 3)),
                MaxPooling2D(pool_size=(2, 2)),
                Conv2D(filters=16, kernel_size=(5, 5), padding='same', activation='relu'),
                MaxPooling2D(pool_size=(2, 2)),
                Flatten(),
                Dense(units=120, activation='relu'),
                Dense(units=84, activation='relu'),
                Dense(units=10, activation='softmax')
			]
		)

def model_fn():
	keras_model = create_keras_model()
	return tff.learning.models.from_keras_model(
			keras_model,
			input_spec = federated_data[0].element_spec,
			loss = tf.keras.losses.SparseCategoricalCrossentropy(),
			metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
		)


training_process = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn,
    client_optimizer_fn = lambda : tf.keras.optimizers.SGD(),
    server_optimizer_fn = lambda : tf.keras.optimizers.SGD()
)

train_state = training_process.initialize()

for i in range(NUM_ROUNDS):
	result = training_process.next(train_state , federated_data)
	train_state = result.state
	train_metrics = result.metrics
	print(f'round {i}, metrics={train_metrics}')


global_weights = train_state.global_model_weights

final_model = tf.keras.Sequential([
    Conv2D(filters=6, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(32 , 32 , 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(filters=16, kernel_size=(5, 5), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(units=120, activation='relu'),
    Dense(units=84, activation='relu'),
    Dense(units=10, activation='softmax')
])

final_model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(), optimizer= tf.keras.optimizers.SGD() , metrics = [tf.keras.metrics.SparseCategoricalAccuracy()])

final_model.set_weights(global_weights[0])

# final_model.evaluate((tf.expand_dims(x_test , axis = -1) , y_test))
test_loss, test_acc = final_model.evaluate(x_test,  y_test, verbose=2)

print('\nTest accuracy:', test_acc)
end_time = datetime.datetime.now()
execution_time = end_time - start_time
print("Program ended at:", end_time)
print("Total execution time:", execution_time)
