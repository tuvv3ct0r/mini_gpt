import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import trange

class Trainer:
    def __init__(self, model, datamodule, config, log_dir='logs', ckpt_dir='checkpoints'):
        self.model = model
        self.datamodule = datamodule
        self.config = config
        self.device = config['device']
        self.writer = SummaryWriter(log_dir=log_dir)
        self.ckpt_dir = ckpt_dir
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config['learning_rate'])
        self.train_losses = []
        self.val_losses = []

    def train(self):
        self.model.to(self.device)
        # Print parameter summary
        print("\nModel Parameter Summary:")
        total_params = 0
        for name, param in self.model.named_parameters():
            print(f"{name}: {param.numel()} params")
            total_params += param.numel()
        print(f"Total parameters: {total_params}\n{'-'*40}")
        best_val_loss = float('inf')
        import random
        for epoch in range(self.config['epochs']):
            self.model.train()
            train_loss = 0
            for batch_idx, (xb, yb) in enumerate(self.datamodule.train_dataloader()):
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits, loss = self.model(xb, yb)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            train_loss /= len(self.datamodule.train_dataloader())
            self.train_losses.append(train_loss)
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            val_loss = self.validate(epoch)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, best=True)
            self.save_checkpoint(epoch, best=False)

            # Randomly sample and generate text after each epoch
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'vocab_size'):
                vocab_size = self.model.config.vocab_size
            else:
                vocab_size = self.config['vocab_size']
            # Get meta for stoi/itos
            meta_path = os.path.join('data', 'meta.pkl')
            if os.path.exists(meta_path):
                import pickle
                with open(meta_path, 'rb') as f:
                    meta = pickle.load(f)
                itos = meta['itos']
                stoi = meta['stoi']
                rand_char = random.choice(list(itos.values()))
                sample_text = self.sample(rand_char, 100, stoi, itos)
                print(f"\n[Sample after epoch {epoch}]\n{sample_text}\n{'-'*40}")
                self.writer.add_text('Sample', sample_text, epoch)
        self.plot_losses()

    def validate(self, epoch):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in self.datamodule.val_dataloader():
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits, loss = self.model(xb, yb)
                val_loss += loss.item()
        val_loss /= len(self.datamodule.val_dataloader())
        self.val_losses.append(val_loss)
        self.writer.add_scalar('Loss/val', val_loss, epoch)
        return val_loss

    def save_checkpoint(self, epoch, best=False):
        ckpt_path = os.path.join(self.ckpt_dir, f"{'best_' if best else ''}model_epoch_{epoch}.pt")
        torch.save(self.model.state_dict(), ckpt_path)

    def plot_losses(self):
        plt.figure()
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.savefig(os.path.join(self.ckpt_dir, 'loss_curve.png'))
        plt.close()

    def sample(self, start_text, length, stoi, itos, temperature=1.0):
        self.model.eval()
        idx = torch.tensor([[stoi[c] for c in start_text]], dtype=torch.long).to(self.device)
        with torch.no_grad():
            for _ in range(length):
                idx_cond = idx[:, -self.model.config.max_seq_len:]
                logits, _ = self.model(idx_cond, None)
                logits = logits[:, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, next_token), dim=1)
        return ''.join([itos[i] for i in idx[0].tolist()]) 