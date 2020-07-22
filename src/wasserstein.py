def generator_w_loss(generator, critic, x):
    '''
    Wassersten loss for generator
    '''
    fake_data = generator.sample(x.shape[0])
    return - critic(fake_data).mean()


def critic_w_loss(generator, critic, x):
    '''
    Wassersten loss for critic (equal to discriminator)
    '''
    fake_data = generator.sample(x.shape[0])
    return critic(fake_data).mean() - critic(x).mean()