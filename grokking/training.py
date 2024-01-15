from math import ceil
import torch
from tqdm import tqdm
import wandb
import numpy as np
import random
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from data import get_data
from model_dropout import Transformer, MLP, LSTMModel
from SAR import Lambda
from Sophia import SophiaG
from sam import SAM
from seng import SENG

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Use the function in your main function or script
set_seed(20010928)

def main(args: dict):
    wandb.init(project="grokking", config=args)
    config = wandb.config
    device = torch.device(config.device)

    # Define time scales
    wandb.define_metric("step")
    wandb.define_metric("epoch")

    # Define metrics
    wandb.define_metric("training/accuracy", step_metric='step')
    wandb.define_metric("training/loss", step_metric='step')
    wandb.define_metric("validation/accuracy", step_metric='epoch')
    wandb.define_metric("validation/loss", step_metric='epoch')
    wandb.define_metric("training/sharpness", step_metric='step')
    wandb.define_metric("validation/sharpness", step_metric='epoch')

    train_loader, val_loader = get_data(
        config.operation,
        config.prime,
        config.training_fraction,
        config.batch_size
        )
    if config.model == 'Transformer':
        model = Transformer(
            num_layers=config.num_layers,
            dim_model=config.dim_model,
            num_heads=config.num_heads,
            num_tokens=config.prime + 2,
            seq_len=5,
            dropout=config.dropout
            ).to(device)
    elif config.model == 'MLP':
        model = MLP(
            num_layers=config.num_layers,
            dim_model=config.dim_model,
            num_heads=config.num_heads,
            num_tokens=config.prime + 2,
            seq_len=5,
            dropout=config.dropout
            ).to(device)
    elif config.model == 'LSTM':
        model = LSTMModel(
            num_layers=config.num_layers,
            dim_model=config.dim_model,
            hidden_dim=config.num_heads * config.dim_model,
            num_tokens=config.prime + 2,
            seq_len=5,
            dropout=config.dropout
        ).to(device)

    preconditioner = None
    if config.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.98),
            weight_decay=config.weight_decay
            )
    elif config.optimizer == 'Sophia':
        optimizer = SophiaG(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.965, 0.99),
            rho = 0.01,
            weight_decay=config.weight_decay)
    elif config.optimizer == 'SAM':
        base_optimizer = torch.optim.SGD  # define an optimizer for the "sharpness-aware" update
        optimizer = SAM(
            model.parameters(),
            base_optimizer,
            lr=config.learning_rate
            # momentum=0.9
            )
    elif config.optimizer == 'SENG':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=0.9
            )
        preconditioner = SENG(model, 1.2, update_freq=200)        

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(config.num_steps/10), gamma=0.8)

    num_epochs = ceil(config.num_steps / len(train_loader))

    for epoch in tqdm(range(num_epochs)):
        train(model, train_loader, optimizer, scheduler, device, config.num_steps, config.optimizer, config.sharp_penalty, preconditioner)
        acc = evaluate(model, val_loader, device, epoch)
        # if acc > 0.999:
        #     break

def train(model, train_loader, optimizer, scheduler, device, num_steps, optimizer_name, sharp_penalty, preconditioner):
    # Set model to training mode
    model.train()
    criterion = torch.nn.CrossEntropyLoss()

    # Loop over each batch from the training set
    for batch in train_loader:

        # Copy data to device if needed
        batch = tuple(t.to(device) for t in batch)

        # Unpack the batch from the loader
        inputs, labels = batch

        # Zero gradient buffers
        optimizer.zero_grad()

        def closure():
            output = model(inputs)[-1,:,:]
            # if sharp_penalty:
            #     loss = criterion(output, labels) + sharp_penalty * Lambda(parameters_to_vector(model.parameters()), model, criterion, inputs, labels)
            # else:
            loss = criterion(output, labels)
            return loss

        # Forward pass
        output = model(inputs)[-1,:,:]
        # sharpness = Lambda(parameters_to_vector(model.parameters()), model, criterion, inputs, labels) / len(labels)
        # wandb.log({"training/sharpness": sharpness.item()})
        # if sharp_penalty:
        #     loss = criterion(output, labels) + sharp_penalty * sharpness
        # else:
        loss = criterion(output, labels)
        acc = (torch.argmax(output, dim=1) == labels).sum() / len(labels)

        # Backward pass
        loss.backward()

        # if optimizer_name == 'SENG':
        #     preconditioner.step()

        # Update weights
        if optimizer_name == 'SAM':
            optimizer.step(closure)
        else:
            optimizer.step()
        scheduler.step()

        metrics = {
            "training/accuracy": acc,
            "training/loss": loss,
            "step": wandb.run.step
        }
        wandb.log(metrics)

        # Finish training at maximum gradient updates
        if wandb.run.step == num_steps:
            return

def evaluate(model, val_loader, device, epoch):
    # Set model to evaluation mode
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    correct = 0
    loss = 0.
    # sharpness = 0.

    # Loop over each batch from the validation set
    for batch in val_loader:
        
        # Copy data to device if needed
        batch = tuple(t.to(device) for t in batch)

        # Unpack the batch from the loader
        inputs, labels = batch
        
        # Forward pass
        with torch.no_grad():
            output = model(inputs)[-1,:,:]
            correct += (torch.argmax(output, dim=1) == labels).sum()
            loss += criterion(output, labels) * len(labels)
            # sharpness += Lambda(parameters_to_vector(model.parameters()), model, criterion, inputs, labels)

    acc = correct / len(val_loader.dataset)
    loss = loss / len(val_loader.dataset)
    # sharpness = sharpness / len(val_loader.dataset)

    metrics = {
        "validation/accuracy": acc,
        "validation/loss": loss,
        "epoch": epoch
        # "validation/sharpness": sharpness.item()
    }
    wandb.log(metrics, commit=False)
    return acc
