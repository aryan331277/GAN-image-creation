dcgan=DCGAN(generator=generator,discriminator=discriminator,latent_dim=LATENT_DIM)
D_LR=0.0001
G_LR=0.0003
dcgan.compile(g_optimizer=Adam(learning_rate=G_LR,beta_1=0.5),d_optimizer=Adam(learning_rate=D_LR,beta_1=0.5),loss_fn=BinaryCrossentropy())
N_EPOCHS=50
dcgan.fit(train_images,epochs=N_EPOCHS,callbacks=[DCGANMonitor()])
