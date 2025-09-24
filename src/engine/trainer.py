import torch
from tqdm.notebook import tqdm
from torch.nn import functional as F
import numpy as np

class TrainerVISKAN:

    def __init__(self, 
                 train_loader, 
                 val_loader, 
                 test_loader, 
                 model, 
                 criterion,
                 optimizer,
                 scheduler,
                 lr, 
                 logger,
                 image_size, 
                 batch_size, 
                 num_epochs, 
                 callback=None,
                 patience=None,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.image_size = image_size
        self.batch_size = batch_size
        self.lr = lr
        self.callback = callback
        self.patience = patience
        self.logger = logger
        self.num_epochs = num_epochs
        self.device = device

        self.logger.info(f"Training initialized with batch_size={batch_size}, lr={lr}, num_epochs={num_epochs}, optimizer={type(optimizer).__name__}, model={type(model).__name__}")


    def train_epoch(self, pbar=None):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels.float())
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            _, labels_max = torch.max(labels.data, 1)
            total += labels.size(0)
            correct += (predicted == labels_max).sum().item()

            if pbar is not None:
                pbar.update(1)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc
    
    def validate_epoch(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels.float())
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                _, labels_max = torch.max(labels.data, 1)
                total += labels.size(0)
                correct += (predicted == labels_max).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc
    
    def test(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels.float())
                
                probs = F.softmax(outputs, dim=1)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                _, labels_max = torch.max(labels.data, 1)
                
                all_probs.append(probs.cpu())
                all_labels.append(labels_max.cpu())
                
                total += labels.size(0)
                correct += (predicted == labels_max).sum().item()
        
        test_loss = running_loss / total
        test_acc = correct / total

        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
        self.logger.info(f"Test results - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")

        all_probs = torch.cat(all_probs).numpy()
        all_labels = torch.cat(all_labels).numpy()

        np.save("logs/logging/test_probs.npy", all_probs)
        np.save("logs/loggingtest_labels.npy", all_labels)
        
    
    def train(self):

        total_iters = self.num_epochs * len(self.train_loader)

        with tqdm(total=total_iters, desc="Training Progress") as pbar:
            for epoch in range(self.num_epochs):
                
                train_loss, train_acc = self.train_epoch(pbar)
                val_loss, val_acc = self.validate_epoch()

                current_lr = self.optimizer.param_groups[0]['lr']

                self.logger.info(
                    f"Epoch {epoch+1}/{self.num_epochs}: "
                    f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                    f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, LR={current_lr:.6f}"
                )

                print(
                    f"Epoch {epoch+1}/{self.num_epochs}: "
                    f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}"
                )

                if self.scheduler:
                    self.scheduler.step()
                
                if self.callback:
                    self.callback(self.model, val_loss, epoch, self.optimizer, self.scheduler)

                if self.callback and self.callback.early_stop:
                    self.logger.info("Early stopping triggered")
                    break

        self.test()

        


    

