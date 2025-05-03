import torch
import torch.nn as nn
from transformers import BertModel
from torchvision import models

class MementoClassifier(nn.Module):
    def __init__(self, num_categories: int, num_tags: int):
        super().__init__()
        
        # Image encoder (using ResNet50)
        resnet = models.resnet50(pretrained=True)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])  # Remove last layer
        
        # Text encoder (using BERT)
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(2048 + 768, 1024),  # 2048 from ResNet, 768 from BERT
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Category classifier
        self.category_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_categories)
        )
        
        # Tag classifier (multi-label)
        self.tag_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_tags),
            nn.Sigmoid()  # For multi-label classification
        )

    def forward(self, images: torch.Tensor, text_inputs: dict) -> tuple:
        # Process images
        img_features = self.image_encoder(images)
        img_features = img_features.view(img_features.size(0), -1)  # Flatten
        
        # Process text
        text_outputs = self.text_encoder(**text_inputs)
        text_features = text_outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Combine features
        combined = self.fusion(torch.cat([img_features, text_features], dim=1))
        
        # Get predictions
        category_logits = self.category_head(combined)
        tag_probs = self.tag_head(combined)
        
        return category_logits, tag_probs

class MementoTrainer:
    def __init__(self, model: MementoClassifier, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        
        # Loss functions
        self.category_criterion = nn.CrossEntropyLoss()
        self.tag_criterion = nn.BCELoss()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5
        )

    def train_step(self, batch: dict) -> dict:
        """Single training step."""
        self.model.train()
        
        # Move batch to device
        images = batch['images'].to(self.device)
        text_inputs = {k: v.to(self.device) for k, v in batch['text_inputs'].items()}
        categories = batch['categories'].to(self.device)
        tags = batch['tags'].to(self.device)
        
        # Forward pass
        category_logits, tag_probs = self.model(images, text_inputs)
        
        # Calculate losses
        category_loss = self.category_criterion(category_logits, categories)
        tag_loss = self.tag_criterion(tag_probs, tags)
        
        # Combined loss
        loss = category_loss + tag_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'category_loss': category_loss.item(),
            'tag_loss': tag_loss.item()
        }

    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> dict:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        total_category_loss = 0
        total_tag_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                images = batch['images'].to(self.device)
                text_inputs = {k: v.to(self.device) for k, v in batch['text_inputs'].items()}
                categories = batch['categories'].to(self.device)
                tags = batch['tags'].to(self.device)
                
                # Forward pass
                category_logits, tag_probs = self.model(images, text_inputs)
                
                # Calculate losses
                category_loss = self.category_criterion(category_logits, categories)
                tag_loss = self.tag_criterion(tag_probs, tags)
                loss = category_loss + tag_loss
                
                total_loss += loss.item()
                total_category_loss += category_loss.item()
                total_tag_loss += tag_loss.item()
        
        num_batches = len(dataloader)
        return {
            'loss': total_loss / num_batches,
            'category_loss': total_category_loss / num_batches,
            'tag_loss': total_tag_loss / num_batches
        } 