from utils import Utils
from models import Models
import numpy as np
import joblib #type:ignore

if __name__ == '__main__':
    # Loading utilities.
    utils = Utils()
    models = Models()
    
    # Loading data.
    data = utils.load_from_csv('./data/felicidad.csv')
    X, y = utils.features_target(data, ['score', 'rank', 'country'], ['score'])
    
    # Retrieve best model.
    models.grid_training(X, y)
    
    # Loading best model to predict a test value.
    model = joblib.load('./models/best_model.pkl') # Loading model
    X_test = np.array([7.594444821,7.479555538,1.616463184,1.53352356,0.796666503,0.635422587,0.362012237,0.315963835,2.277026653])
    prediction = model.predict(X_test.reshape(1,-1))
    print(prediction)