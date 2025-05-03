import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import logging
from datetime import datetime

from data_loader import MementoDataLoader
from model import MementoClassifier, MementoTrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_pipeline/training.log'),
        logging.StreamHandler()
    ]
)

class MementoDataset(Dataset):
    def __init__(self, data: dict):
        self.images = data['images']
        self.texts = data['texts']
        self.categories = data['categories']
        self.tags = data['tags']

    def __len__(self):
        return len(self.categories)

    def __getitem__(self, idx):
        return {
            'images': self.images[idx],
            'text_inputs': self.texts[idx],
            'categories': self.categories[idx],
            'tags': self.tags[idx]
        }

def load_categories_and_tags():
    """Load category and tag information."""
    with open('memento_categories.json', 'r') as f:
        categories = json.load(f)
    with open('memento_tags.json', 'r') as f:
        tags = json.load(f)
    return categories, tags

def main():
    # Load configuration
    categories, tags = load_categories_and_tags()
    num_categories = len(categories)
    num_tags = len(tags)
    
    # Initialize data loader
    data_loader = MementoDataLoader('firebase-credentials.json')
    
    # Load training data
    logging.info("Loading training data...")
    training_mementos = data_loader.load_training_data()
    
    # Prepare training data
    logging.info("Preparing training data...")
    training_data = data_loader.prepare_training_data(training_mementos)
    
    # Create dataset and dataloader
    dataset = MementoDataset(training_data)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize model and trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MementoClassifier(num_categories, num_tags)
    trainer = MementoTrainer(model, device)
    
    # Training loop
    num_epochs = 10
    best_loss = float('inf')
    
    logging.info(f"Starting training on {device}...")
    for epoch in range(num_epochs):
        # Training
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            loss_dict = trainer.train_step(batch)
            epoch_loss += loss_dict['loss']
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        logging.info(f"Epoch {epoch + 1}/{num_epochs} - Average Loss: {avg_epoch_loss:.4f}")
        
        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'loss': best_loss,
            }, 'ml_pipeline/best_model.pth')
            logging.info(f"Saved new best model with loss: {best_loss:.4f}")
    
    logging.info("Training completed!")

if __name__ == "__main__":
    main() 