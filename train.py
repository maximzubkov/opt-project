from collections import defaultdict

import torch
from IPython.display import clear_output
from tqdm import tqdm_notebook

import utils.pytorch_utils as ptu


def train(
    generator,
    critic,
    c_loss_fn,
    g_loss_fn,
    train_loader,
    g_optimizer,
    c_optimizer,
    n_critic=1,
    g_scheduler=None,
    c_scheduler=None,
):
    g_losses, g_grad, c_grad, c_losses = [], [], [], []
    generator.train(True)
    critic.train(True)
    for i, x in enumerate(train_loader):
        x = x.to(ptu.device).float()

        c_loss = c_loss_fn(generator, critic, x)
        c_optimizer.zero_grad()
        c_loss.backward()
        if i % 300 == 0:
            c_grad.append(0)
            for param in critic.parameters():
                if param.requires_grad:
                    c_grad[-1] += torch.norm(param.grad.data).detach().numpy() ** 2
        c_optimizer.step()

        if i % n_critic == 0:
            g_loss = g_loss_fn(generator, critic, x)
            g_optimizer.zero_grad()
            g_loss.backward()
            if i % 300 == 0:
                g_grad.append(0)
                for param in generator.parameters():
                    if param.requires_grad:
                        g_grad[-1] += torch.norm(param.grad.data).detach().numpy() ** 2
            g_optimizer.step()

            if g_scheduler is not None:
                g_scheduler.step()
            if c_scheduler is not None:
                c_scheduler.step()
    return dict(g_losses=g_losses, c_losses=c_losses, g_grad=g_grad, c_grad=c_grad)


def train_epochs(
    experiment,
    generator,
    critic,
    g_loss_fn,
    c_loss_fn,
    train_loader,
    train_args,
    g_opt,
    c_opt,
    g_scheduler=None,
    c_scheduler=None,
    name="",
    verbose=False,
):
    epochs = train_args["epochs"]

    train_logs = defaultdict(list)

    for epoch in tqdm_notebook(range(epochs), desc="Epoch", leave=False):
        if epoch == 0:
            start_snapshot = get_training_snapshot(generator)
        generator.train(True)
        critic.train(True)
        train_loss = train(
            generator,
            critic,
            c_loss_fn,
            g_loss_fn,
            train_loader,
            g_opt,
            c_opt,
            n_critic=train_args.get("n_critic", 1),
            g_scheduler=g_scheduler,
            c_scheduler=c_scheduler,
        )

        for k in train_loss:
            train_logs[k].extend(train_loss[k])

        evaluation_results = experiment.eval(generator, critic)
        for k in evaluation_results:
            train_logs[k].append(evaluation_results[k])
        if (epoch % 2 == 0) and verbose:
            clear_output(wait=True)
            experiment.epoch_vizual(train_logs, path=f"results/{name}/output_{epoch}_tmp.pdf")

    if train_args.get("final_snapshot", False):
        final_snapshot = get_training_snapshot(generator)
        return (train_logs, start_snapshot, final_snapshot)
    else:
        return train_logs


def get_training_snapshot(generator, n_samples=10000):
    with torch.no_grad():
        samples = ptu.get_numpy(generator.sample(n_samples))
        return samples
