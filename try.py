from tensorflow import keras
from keras.models import load_model
import h5py
model = load_model('model.h5', compile = False) 
model.summary()