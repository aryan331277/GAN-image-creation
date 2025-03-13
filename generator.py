#generator model creates new image from training data from noise
model=Sequential(name="generator")
model.add(layers.Dense(8*8*512,input_dim=LATENT_DIM))#creating noise which is 1d
model.add(layers.ReLU())

model.add(layers.Reshape((8,8,512)))#1d to 3d

#upsmple to 16*16
model.add(layers.Conv2DTranspose(256,(4,4),strides=(2,2),padding="same",kernel_initializer=WEIGHT_INIT))
model.add(layers.ReLU())

#upsmple to 32*32
model.add(layers.Conv2DTranspose(128,(4,4),strides=(2,2),padding="same",kernel_initializer=WEIGHT_INIT))
model.add(layers.ReLU())

model.add(layers.Conv2DTranspose(64,(4,4),strides=(2,2),padding="same",kernel_initializer=WEIGHT_INIT))
model.add(layers.ReLU())


model.add(layers.Conv2D(CHANNELS,(4,4),padding="same",activation="tanh"))

generator=model
generator.summary()
