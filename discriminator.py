model=Sequential(name="discriminator")
input_shape=(64,64,3)
alpha=0.2

model.add(layers.Conv2D(64,(4,4),strides=(2,2),padding="same",input_shape=input_shape))
model.add(layers.LeakyReLU(alpha=alpha))

model.add(layers.Conv2D(128,(4,4),strides=(2,2),padding="same",input_shape=input_shape))
model.add(layers.LeakyReLU(alpha=alpha))

model.add(layers.Conv2D(128,(4,4),strides=(2,2),padding="same",input_shape=input_shape))
model.add(layers.LeakyReLU(alpha=alpha))

model.add(layers.Flatten())
model.add(layers.Dropout(0.3))

model.add(layers.Dense(1,activation="sigmoid"))

discriminator=model
discriminator.summary()
