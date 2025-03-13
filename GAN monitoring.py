import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

class DCGANMonitor(keras.callbacks.Callback):
    def __init__(self, num_imgs=25, latent_dim=100):
        self.num_imgs = num_imgs
        self.latent_dim = latent_dim
        self.noise = tf.random.normal([self.num_imgs, latent_dim])  

    def on_epoch_end(self, epoch, logs=None):
        g_img = self.model.generator(self.noise, training=False)  
        g_img = (g_img + 1) * 127.5  
        g_img = tf.clip_by_value(g_img, 0, 255)  
        g_img = g_img.numpy().astype("uint8")  
        
        fig = plt.figure(figsize=(8, 8))
        for i in range(self.num_imgs):
            plt.subplot(5, 5, i + 1)
            plt.imshow(g_img[i])  
            plt.axis("off")

        plt.show()

    def on_train_end(self, logs=None):
        self.model.generator.save("generator.h5")  
