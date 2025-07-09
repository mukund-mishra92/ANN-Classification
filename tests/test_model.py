import tensorflow as tf
#from src import load_model  # example function

def test_model_loading():
    model = tf.keras.models.load_model('src/customer_churn_model.h5')
    #model = load_model()
    assert model is not None
