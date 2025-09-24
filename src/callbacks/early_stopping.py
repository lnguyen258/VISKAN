import torch


class EarlyStopping:
    def __init__(self, patience, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, model, val_loss, epoch, optimizer=None, scheduler=None):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model, val_loss, epoch, optimizer, scheduler)
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model, val_loss, epoch, optimizer, scheduler)
            self.counter = 0  

    def save_checkpoint(self, model, val_loss, epoch, optimizer, scheduler):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss': val_loss
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        torch.save(checkpoint, 'logs/checkpoint/checkpoint_01.pt')

        if self.verbose:
            print(f"Saved checkpoint at epoch {epoch} with val_loss {val_loss}")