def generator_w_loss(generator, discriminator, x):
    """
    Wassersten loss for generator
    """
    fake_data = generator.sample(x.shape[0])
    return -discriminator(fake_data).mean()


def discriminator_w_loss(generator, discriminator, x):
    """
    Wassersten loss for discriminator
    """
    fake_data = generator.sample(x.shape[0])
    return discriminator(fake_data).mean() - discriminator(x).mean()
