import flwr as fl
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, MaxPooling2D


def create_new_model():
    model = tf.keras.Sequential([
    Conv2D(filters=6, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(32 , 32 , 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(filters=16, kernel_size=(5, 5), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(units=120, activation='relu'),
    Dense(units=84, activation='relu'),
    Dense(units=10, activation='softmax')
  ])
    
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def evaluate_fn(server_round , parameters_ndarrays , someList):
    (x_train , y_train) , (x_test , y_test) = tf.keras.datasets.cifar10.load_data()
    model = create_new_model()
    model.set_weights(parameters_ndarrays)
    test_loss , test_accuracy = model.evaluate(x_test , y_test, batch_size=32)
    model.save('flower_final_model.h5')
    file_path = 'loss_metrics.txt' 
    with open(file_path, 'w') as file:
        file.write(f'server_round = {server_round}\n')
        file.write('test_loss : {} , test_accuracy : {}\n'.format(test_loss , test_accuracy))
    test_metrics = {'accuracy' : test_accuracy}
    return test_loss , test_metrics

strategy = fl.server.strategy.FedAvg(
    evaluate_fn = evaluate_fn
)

history = fl.server.start_server(config=fl.server.ServerConfig(num_rounds=5), strategy=strategy)

file_path = 'history.txt'
with open(file_path, 'w') as file:
        file.write(repr(history))

