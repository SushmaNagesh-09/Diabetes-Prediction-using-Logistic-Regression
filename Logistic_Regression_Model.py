# Importing dependencies

import numpy as np 

class Logistic_Regression():
    
    def __init__(self, learning_rate, no_of_iterations):
        
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations
        
    
    # fit function to train the model with the dataset 
    def fit(self, X, Y):
        
        # number od data point in the dataset(number of rows) --> m
        # number of input features in the dataset --> n
        self.m , self.n = X.shape      
        
        
        # initiating weight and bias
        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        # implementing Gradient Descent for Optimization
        for i in range(self.no_of_iterations):
            self.update_weights()
            
             
    def update_weights(self ):
        # Y_hat formula(sigmoid function)
        Y_hat = 1 / (1 + np.exp(- ( self.X.dot (self.w) + self.b )))   #  z = wX + b
        
        # derivatives
        dw = (1 / self.m) * np.dot(self.X.T, (Y_hat - self.Y ))
        
        db = (1 / self.m) * np.sum(Y_hat - self.Y)
 
        # updates weights and bias using Gradient descent algo
        self.w = self.w - self.learning_rate * dw
        
        self.b = self.b - self.learning_rate * db
        
    
    # Sigmoid Equation & Decision Boundary
    def predict(self, x):
        Y_pred = 1 / (1 + np.exp(- ( x.dot (self.w) + self.b )))
        
        Y_pred = np.where(Y_pred > 0.5, 1, 0)
        return Y_pred
        