dcgan=DCGAN(generator=generator,discriminator=discriminator,latent_dim=LATENT_DIM)
D_LR = 2e-4  
G_LR = 1e-4  
dcgan.compile(
    g_optimizer=Adam(learning_rate=G_LR, beta_1=0.5), 
    d_optimizer=Adam(learning_rate=D_LR, beta_1=0.5), 
    loss_fn=BinaryCrossentropy(from_logits=True)
)

# Train with Monitoring
N_EPOCHS = 200
dcgan.fit(train_images, epochs=N_EPOCHS, callbacks=[DCGANMonitor()])
