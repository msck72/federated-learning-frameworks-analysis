import tensorflow as tf
import json

def load_data():
    with open("num_clients.json", "r") as file:
        stored_data = json.load(file)

    (x_train , y_train) , (x_test , y_test) = tf.keras.datasets.cifar10.load_data()
    num_of_clients = stored_data["num_clients"]
    start =  stored_data["start"]
    end = stored_data["start"] + stored_data["each_client_data"]
    stored_data["num_clients"] = num_of_clients + 1 
    stored_data["start"] = end 

    # Writing modified data back to the file
    with open("num_clients.json", "w") as file:
        json.dump(stored_data, file)

    return (x_train[start : end], y_train[start : end]) , (x_test , y_test)
   
        