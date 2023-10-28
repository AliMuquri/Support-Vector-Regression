import tensorflow as tf
from tensorflow.keras.losses import Loss


class SupportVectorLoss(Loss):
    ''' loss function used for supporvector regression'''
    def __init__(self,  reduction=tf.keras.losses.Reduction.AUTO, name=None, epsilon=1.0):
        ''' initialize  constructor of the parent and sets epsilon 
        Args:
        epsilon: controls the width of the margin
        Returns:
        class object '''
        super(SupportVectorLoss, self).__init__()
        self.epsilon = epsilon

    def call(self, y_true, y_pred):
        '''computes the huber loss 
        Args:
        y_true: the groundtruth
        y_pred: model prediction
        Returns:
        output: huber loss
        '''
      
     
        #output = tf.keras.losses.Huber(self.epsilon)(tf.math.log(tf.maximum(y_true,1)), tf.math.log(tf.maximum(y_pred,1)))
        output = tf.math.log(tf.keras.losses.Huber(self.epsilon)(y_true, y_pred) + 1)
        
        return output
