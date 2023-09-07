import tensorflow as tf
from keras.models import load_model

def cargarModelo():

    FILENAME_MODEL_TO_LOAD = "covid19_model_full.h5"

    # Cargar la RNA desde disco
    loaded_model = load_model(FILENAME_MODEL_TO_LOAD)
    print("Modelo cargado de disco << ", loaded_model)

    graph = tf.compat.v1.get_default_graph()
    return loaded_model, graph
