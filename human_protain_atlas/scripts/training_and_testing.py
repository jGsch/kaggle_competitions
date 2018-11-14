import torch
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_dataloader):
    """ Train a model over all data
    """

    # Set model to training mode
    model.net.train()

    for inputs, labels, _ in train_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        model.optimizer.zero_grad()

        # forward
        outputs = model.net(inputs)
        _, preds = torch.max(outputs, 1)
        loss = model.criterion(outputs, labels)

        # backward
        loss.backward()
        model.optimizer.step()

        # update running statistics
        model.score.update('train', loss, inputs, preds, labels)

    # Update epoch training statistics
    model.score.update_scores('train')

    del inputs, labels, outputs, preds


def valid(model, valid_dataloader):
    """  Valid a model over all data
    """

    # Set model to evaluation mode
    model.net.eval()

    worst_score = 100

    for inputs, labels, path in valid_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        outputs = model.net(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = model.criterion(outputs, labels)

        # Update running statistics
        model.score.update('valid', loss, inputs, preds, labels)

        if model.score.running_score['valid'] <= worst_score:
            worst_score = model.score.running_score['valid']
            worst_score_path = path[0]
            # print('Worst score %.5f' % model.score.running_score['valid'])
            # print(",\n" . join(path[0]))

    # Update epoch testing statistics
    model.score.update_scores('valid', (worst_score, worst_score_path))

    del inputs, labels, outputs, preds




def train_model(model, dataloaders, config):
    """  Train and valid a model over all data for a certain number of epochs
    """

    if model.score.verbose:
        print('-'*101 + '\nTraining\n' + ' '*91 + '('+time.strftime('%X')+')')

    # Start timer
    model.score.time_counter('start')

    for epoch in range(config['n_epochs']):
        if model.score.verbose:
            print('Epoch %2d/%2d:' % (epoch+1, config['n_epochs']), end=' '*3)

        # Training
        train(model, dataloaders['train'])

        # Validation
        valid(model, dataloaders['valid'])

        # Keep best model
        if model.score.score['valid'][-1] >= model.score.best_score['valid']:
            model.save_best_weights()

        # Update learning rate scheduler
        metric = model.score.best_score['valid'] if isinstance(model.scheduler, ReduceLROnPlateau) else None
        model.scheduler.step(metric)

        # Print loss and scores for each epoch
        model.score.print_epoch_scores(epoch)

        # Save checkpoint
        model.save(epoch=epoch, checkpoint=True)

    # Stop timer, load best weights and print scores
    model.score.time_counter('stop', phase='training')
    model.load_best_weights()
    model.score.print_scores('training')


def test_model(model, dataloaders):
    """  Test a model over all data
    """

    if model.score.verbose:
        print('-'*105 + '\nTesting\n')

    # Set model to evaluation mode
    model.net.eval()

    # Start perf_counter for testing phase
    model.score.time_counter('start')

    for inputs, labels, _ in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward
        outputs = model.net(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = model.criterion(outputs, labels)

        # Update running statistics
        model.score.update('test', loss, inputs, preds, labels)

    # Stop perf_counter, update scores and print them
    model.score.time_counter('stop', phase='testing')
    model.score.update_scores('test')
    model.score.print_scores('testing')
