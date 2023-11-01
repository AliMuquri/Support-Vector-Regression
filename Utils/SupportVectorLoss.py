import tensorflow as tf
from tensorflow.keras.losses import Loss
import re


class SupportVectorLoss(Loss):
    ''' loss function used for supporvector regression'''
    def __init__(self, lambda_= 1e-3 , weights=None,  reduction=tf.keras.losses.Reduction.AUTO, name=None, epsilon=1):
        ''' initialize  constructor of the parent and sets epsilon 
        Args:
        epsilon: controls the width of the margin,
        lambda_: regularization strength
        weights: the model weights
        Returns:
        class object '''
        super(SupportVectorLoss, self).__init__()
        self.epsilon = epsilon
        self.regularization = False
       

        if type(weights) != type(None):
            self.weights = [var for var in weights if re.match(r'alpha:\d+', var.name)]
            self.regularization = True
        self.lambda_ = lambda_
         
    def call(self, y_true, y_pred):
        '''computes the huber loss 
        Args:
        y_true: the groundtruth
        y_pred: model prediction
        Returns:
        total_lsos: huber loss and regularization
        '''
      
     
        loss = tf.keras.losses.Huber(self.epsilon)(y_true, y_pred)
   
        total_loss = loss + self.lambda_ * tf.reduce_sum(tf.convert_to_tensor([tf.square(w) for w in self.weights])) if self.regularization else loss
       

        return total_loss
