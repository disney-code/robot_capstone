from tensorflow import keras
import tensorflow as tf
import numpy as np



class FGSM:
    """
    We use FGSM to generate a batch of adversarial examples. 
    """
    def __init__(self, model, ep=0.01, isRand=True):
        """
        isRand is set True to improve the attack success rate. 
        """
        self.isRand = isRand
        self.model = model
        self.ep = ep
        
    def generate(self, x, y, randRate=1):
        """
        x: clean inputs, shape of x: [batch_size, width, height, channel] 
        y: ground truth, one hot vectors, shape of y: [batch_size, N_classes] 
        """
        fols = []
        target = tf.constant(y)
        
        xi = x.copy()
        if self.isRand:
            x = x + np.random.uniform(-self.ep * randRate, self.ep * randRate, x.shape)
            x = np.clip(x, 0, 1)
        
        x = tf.Variable(x)
        grads=[]
        count=0
        for i in range(int(x.shape[0]/500)):
            print(f"FGSM batch in x: {i}")
            with tf.GradientTape() as tape:
                x_=tf.Variable(x[count:count+500])
                target_=tf.constant(y[count:count+500])
                loss = keras.losses.categorical_crossentropy(target_, self.model(x_))
                grads.append(tape.gradient(loss, x_))
            del tape
        
        grads = tf.concat(grads,axis=0)
        delta = tf.sign(grads)
        x_adv = x + self.ep * delta
        
        x_adv = tf.clip_by_value(x_adv, clip_value_min=xi-self.ep, clip_value_max=xi+self.ep)
        x_adv = tf.clip_by_value(x_adv, clip_value_min=0, clip_value_max=1)
        
        idxs = np.where(np.argmax(self.model(x_adv), axis=1) != np.argmax(y, axis=1))[0]
        print("SUCCESS:", len(idxs))
        
        x_adv, xi, target = x_adv.numpy()[idxs], xi[idxs], target.numpy()[idxs]
        x_adv, target = tf.Variable(x_adv), tf.constant(target)
        
        preds = self.model(x_adv).numpy()
        ginis = np.sum(np.square(preds), axis=1)
        
        with tf.GradientTape() as tape:
            loss = keras.losses.categorical_crossentropy(target, self.model(x_adv))
            grads = tape.gradient(loss, x_adv)
            grad_norm = np.linalg.norm(grads.numpy().reshape(x_adv.shape[0], -1), ord=1, axis=1)
            grads_flat = grads.numpy().reshape(x_adv.shape[0], -1)
            diff = (x_adv.numpy() - xi).reshape(x_adv.shape[0], -1)
            for i in range(x_adv.shape[0]):
                i_fol = -np.dot(grads_flat[i], diff[i]) + self.ep * grad_norm[i]
                fols.append(i_fol)
  
        return x_adv.numpy(), target.numpy(), np.array(fols), ginis



class PGD:
    """
    We use PGD to generate a batch of adversarial examples. PGD could be seen as iterative version of FGSM.
    """
    def __init__(self, model, ep=0.01, step=None, epochs=10, isRand=True):
        """
        isRand is set True to improve the attack success rate. 
        """
        self.isRand = isRand
        self.model = model
        self.ep = ep
        if step == None:
            self.step = ep/6
        self.epochs = epochs
        
    def generate(self, x, y, randRate=1):
        """
        x: clean inputs, shape of x: [batch_size, width, height, channel] 
        y: ground truth, one hot vectors, shape of y: [batch_size, N_classes] 
        """
        fols = []
        target = tf.constant(y)
    
        xi = x.copy()
        if self.isRand:
            x = x + np.random.uniform(-self.ep * randRate, self.ep * randRate, x.shape)
            x = np.clip(x, 0, 1)
        
        grads_list=[]
        count=0
        
        x_adv = tf.Variable(x)
        for i in range(self.epochs):
            for j in range(int((x.shape[0])/500)):
                with tf.GradientTape() as tape:
                    x_=tf.Variable(x_adv[count:count+500])
                    target_=tf.constant(target[count:count+500])
                    loss = keras.losses.categorical_crossentropy(target_, self.model(x_))
                    grads_list.append(tape.gradient(loss, x_))
                    count+=500
                del tape
                
            grads = tf.concat(grads_list,axis=0)    
            delta = tf.sign(grads)
            x_adv.assign_add(self.step * delta)
            x_adv = tf.clip_by_value(x_adv, clip_value_min=xi-self.ep, clip_value_max=xi+self.ep)
            x_adv = tf.clip_by_value(x_adv, clip_value_min=0, clip_value_max=1)
            x_adv = tf.Variable(x_adv)
        
        idxs = np.where(np.argmax(self.model(x_adv), axis=1) != np.argmax(y, axis=1))[0]
        print("SUCCESS:", len(idxs))
        
        x_adv, xi, target = x_adv.numpy()[idxs], xi[idxs], target.numpy()[idxs]
        print("line 127")
        x_adv, target = tf.Variable(x_adv), tf.constant(target)
        print("line 129")
        preds = self.model(x_adv).numpy()
        print("line 131")
        ginis = np.sum(np.square(preds), axis=1)
        print("line 133")
        count=0
        grads=[]
        for i in range(int(x_adv.shape[0]/500)):
            print(f"line 135 i: {i}")
            with tf.GradientTape() as tape:
                x_=tf.Variable(x_adv[count:count+500])
                target_=tf.constant(target[count:count+500])
                loss = keras.losses.categorical_crossentropy(target_, self.model(x_))
                print("line 142")
                grads.append(tape.gradient(loss, x_))
                count+=500
            del tape
        with tf.GradientTape() as tape:
            print(f"line 147 count stopped at={count}")
            x_=tf.Variable(x_adv[count:])
            target_=tf.constant(target[count:])
            loss = keras.losses.categorical_crossentropy(target_, self.model(x_))
            grads.append(tape.gradient(loss, x_))
        print("line 152")
        print("type of grads: ",type(grads))#list
        print("type of grads[0]: ",type(grads[0]))
        print("shape of grads[0]: ", grads[0].shape)
        print("length of grads: ",len(grads))
        #print("grads shape: ",grads.shape)
        grads = tf.concat(grads,axis=0)
        print("grads.numpy().shape: ",grads.numpy().shape) #(33500,32,32,3)
        print("grads calculated successfully")
        print(f"x_adv.shape: {x_adv.shape}")  #(33979, 32, 32, 3)
        grad_norm = np.linalg.norm(grads.numpy().reshape(x_adv.shape[0], -1), ord=1, axis=1)
        grad_norm = np.linalg.norm(grads.numpy().reshape(x_adv.shape[0], -1), ord=1, axis=1)
        print("line 149")
        grads_flat = grads.numpy().reshape(x_adv.shape[0], -1)
        diff = (x_adv.numpy() - xi).reshape(x_adv.shape[0], -1)
        for i in range(x_adv.shape[0]):
            print(f"i = {i}/{x_adv.shape[0]}")
            i_fol = -np.dot(grads_flat[i], diff[i]) + self.ep * grad_norm[i]
            fols.append(i_fol)
  
        return x_adv.numpy(), target.numpy(), np.array(fols), ginis
    

  
    