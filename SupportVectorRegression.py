import tensorflow as tf
from tensorflow import keras
from tensorflow import Module
from Utils import SupportVectorLoss
from tensorflow.keras.models import Model
from tensorflow import Module
from tensorflow.keras.callbacks import ProgbarLogger, ModelCheckpoint

@keras.utils.register_keras_serializable(name='SupporVectorComponent')
class SupportVectorComponent(Module):
    ''' Provide the functionality for SVR '''

    def __init__(self, input_dim, sigma=10):
        ''' initilize the weights 
        Args:
        input_dim: the number of features in the data
        sigma: controls the width of the kernel
        Returns:
        class object'''
        self.input_dim = input_dim
        self.alpha = tf.Variable(initial_value=tf.random.normal(shape=(self.input_dim,1)), name='alpha')
        self.bias = tf.Variable(initial_value= tf.zeros(shape=(1,1)), name='bias')
        self.sigma  =  sigma


    @tf.function
    def kernel_rbf(self, features):
        ''' computes the kernel value
        Args:
        features: the data features
        Returns:
        kernel: the kernel value
        '''
        pairwise_distance = tf.reduce_sum(tf.square(features[:, tf.newaxis] - features), axis=2)
        kernel = tf.exp(-pairwise_distance /(2*self.sigma**2))       
        return kernel

    def __call__(self, inputs):
        ''' compute the feedforward
        Args:
        inputs: the data
        Returns:i
        output: the computed feedforward  '''
        kernel_matrix = self.kernel_rbf(inputs)
     
        output = tf.matmul(kernel_matrix, self.alpha) + self.bias
       
        return output

    def get_config(self):
        ''' adpat config for saving
        Returns:
        config: congif dict
        '''
        config = {'input_dim': self.input_dim, 'sigma': self.sigma}
        return config

    @classmethod
    def from_config(cls, config):
        '''creates an instance of the model from this config
        Returns
        instance:an instane of the model '''
        instance = cls(**config)
        return instance

@keras.utils.register_keras_serializable(name='SupportVectorRegression')
class SupportVectorRegression(Model):
    ''' Performs train and predict of support vector regression'''

    def __init__(self, input_dim, sigma=5):
        ''' initilize the support vector component
        Args:
        input_dim: the number of features in the data
        epsilon: controls teh weidth of the margin
        Returns:
        class object '''
        super(SupportVectorRegression, self).__init__()
        self.input_dim = input_dim
        self.sigma = sigma
        self.sv_component = SupportVectorComponent(self.input_dim, self.sigma)


    @tf.function 
    def train_step(self, data):
        ''' custom train_step where both features and labels are passed on
        Args:
        data: the features and labels
        Returns:
        results: a dictionary containing the results of the training_step '''

        y_pred, loss = None, None
        x, y = data
        
        with tf.GradientTape() as tape:
            y_pred =  self(x)
            loss = self.compiled_loss(y, y_pred)
           
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y,y_pred)

        results = {m.name: m.result() for m in self.metrics}
        return results

    @tf.function
    def validation_step(self, data):
        ''' a custom validation steo where both feature sand labels are passed on
        Args:
        data: the features and label
        Returns:
        results a dictionary containing teh results of the validation step'''

        x, y  = data
        y_pred = self(x)
        loss = self.compiled_loss(y, y_pred)
        self.compiled_metrics.update_state(y, y_pred)

        results = {"val_" + m.name: m.result() for m in self.metrics}


        return results


    def fit(self,train_data,epochs, steps_per_epoch=None, validation_data=None, callbacks=None):
        ''' a customized fit method where the model training progresses
        Args:
        train_data: the training data
        epochs: the number of epochs the training will execute
        validation_data: the validation data
        callbacks: all callback instance that being used during this training
        Returns:
        None '''
        if type(steps_per_epoch)==type(None):
            steps_per_epoch = sum([1 for b in train_data])
      

        #define and initialize the progbar
        progbar = ProgbarLogger(count_mode='steps')
        progbar.target = steps_per_epoch
 
        #if save_freq is not an epoch it needs to be handled on a batch level
        modelcheckpoint = next((callback for callback in callbacks if isinstance(callback, ModelCheckpoint)), None)

        #if callback has not been defined
        if type(callbacks) == type(None):
            callbacks = []

        #Instead of passing the callback make sure it is always included,
        #Use the callbacks and not utils for progbar. Utils version fail on graph mode.
        default_callbacks= [progbar]
        callbacks = callbacks + default_callbacks

        #associate the call back with the model
        for callback in callbacks:
            callback.set_model(self)

        def compute_mean(results):
            ''' computes the mean value of the results  (list of tuples)
            Args:
            results: a tensor of tuples
            Returns:
            mean: the mean in list of tuples
            mean_dict: the mean in dict'''
            size = len(results)
            computed_dict = {}
            for result in results:
                for key in result.keys():
                    try:
                        computed_dict[key] += result[key] / size
                    except:
                        computed_dict[key] = result[key] / size
            mean_dict = computed_dict
            return mean_dict

        def train_progress(index, data):
            ''' computes the forwardpass and updates the progressbar
            Args:
            index: the current training step in the epoch
            data: the dataset
            Returns:
            results: the results of the train step
            '''
            results_dict = self.train_step(data)
            progbar.on_train_batch_end(0, logs=results_dict)

            #incase of using batch saving in modelcheckpoint
            if type(modelcheckpoint)!= type(None) and isinstance(modelcheckpoint.save_freq, int):
                #Avoid massive computational cost, adapt the code from _should-save_one_batch
               #print("last save {}".format(modelcheckpoint._batches_seen_since_last_saving)) 
               # print("save_Freq {}".format(modelcheckpoint.save_freq))
                if (modelcheckpoint._batches_seen_since_last_saving) >= modelcheckpoint.save_freq-1:
                   
                    validation_results = [self.validation_step(batch) for batch in validation_data]
                    mean_validation_dict = compute_mean(validation_results)
                    modelcheckpoint.on_train_batch_end(index,logs=mean_validation_dict)
                else:
                   
                    modelcheckpoint.on_train_batch_end(index, logs={'val_loss': 0})
            return results_dict

       
        for epoch in tf.range(epochs):
            
            modelcheckpoint.on_epoch_begin(epoch)

            if epoch%10==0:
                print("Epoch: {} / {}".format(epoch, epochs))

            mean_train_dict, mean_validation_dict = None, None

            train_results = [train_progress(index, batch) for index,batch in enumerate(train_data.take(steps_per_epoch))]
            mean_train_dict = compute_mean(train_results)

            if type(None)!=type(validation_data):

                validation_results = [self.validation_step(batch) for batch in validation_data]
                mean_validation_dict = compute_mean(validation_results)

            #if validation is provided it merges both training and validation results
            condition = type(mean_train_dict)!=type(None) and type(mean_validation_dict)!=type(None) 
            merged_dict = {**mean_train_dict, **mean_validation_dict } if condition else mean_train_dict

            #updates the calls backs
            for callback in callbacks: 
                callback.on_epoch_end(tf.cast(epoch,dtype=tf.int64), logs=merged_dict)

    def call(self, inputs):
        ''' forward pass of the model 
        Args:
        inputs: the data
        Returns:
        output: the computed forward pass '''
        output= self.sv_component(inputs)
        return output

    def predict(self, inputs):
        ''' make a prediction
        Args.
        inputs: the data
        Returns:
        predict: the computed  modle prediction
        '''
        prediction = self(inputs)
        return prediction

    def get_config(self):
        ''' adapt congif for saving
        Returns:
        config: congif dict
        '''
        config = {'input_dim': self.input_dim, 'sigma': self.sigma}
        return config
    
    @classmethod
    def from_config(cls, config):
        '''build isntance from config,
        Returns:
        instance: an model instance'''
        
     
        instance =  cls(**config)
        return instance
                   
