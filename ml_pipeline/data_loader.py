import json
import os
from typing import List, Dict, Any
import firebase_admin
from firebase_admin import credentials, firestore
from PIL import Image
import torch
from transformers import BertTokenizer, BertModel
from torchvision import transforms
from tqdm import tqdm

class MementoDataLoader:
    def __init__(self, firebase_credentials_path: str):
        """Initialize the data loader with Firebase credentials."""
        # Initialize Firebase
        cred = credentials.Certificate(firebase_credentials_path)
        firebase_admin.initialize_app(cred)
        self.db = firestore.client()
        
        # Initialize BERT tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        
        # Image transformation pipeline
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])

    def load_training_data(self) -> List[Dict[str, Any]]:
        """Load user mementos from Firebase for training."""
        mementos = []
        
        # Load user mementos
        user_mementos = self.db.collection('mementos').stream()
        for doc in tqdm(user_mementos, desc="Loading user mementos"):
            memento = doc.to_dict()
            memento['id'] = doc.id
            mementos.append(memento)
            
        return mementos

    def load_public_mementos(self, public_data_dir: str) -> List[Dict[str, Any]]:
        """Load public mementos from JSON files."""
        public_mementos = []
        
        # Walk through all subdirectories
        for root, _, files in os.walk(public_data_dir):
            for file in files:
                if file.endswith('.json'):
                    with open(os.path.join(root, file), 'r') as f:
                        mementos = json.load(f)
                        if isinstance(mementos, list):
                            public_mementos.extend(mementos)
                        else:
                            public_mementos.append(mementos)
        
        return public_mementos

    def process_text(self, text: str) -> torch.Tensor:
        """Process text using BERT."""
        inputs = self.tokenizer(text, return_tensors="pt", 
                              padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)  # [CLS] token embedding

    def process_image(self, image_path: str) -> torch.Tensor:
        """Process image using standard image transformation pipeline."""
        try:
            image = Image.open(image_path).convert('RGB')
            return self.image_transform(image)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return torch.zeros((3, 224, 224))  # Return zero tensor for failed images

    def prepare_training_data(self, mementos: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Prepare training data from mementos."""
        texts = []
        images = []
        categories = []
        tags = []
        
        for memento in tqdm(mementos, desc="Preparing training data"):
            # Process text
            text = f"{memento.get('name', '')} {memento.get('description', '')}"
            text_features = self.process_text(text)
            texts.append(text_features)
            
            # Process image if available
            if memento.get('media') and len(memento['media']) > 0:
                image_features = self.process_image(memento['media'][0])
                images.append(image_features)
            else:
                images.append(torch.zeros((3, 224, 224)))
            
            # Process categories and tags
            categories.append(memento.get('category', 'other'))
            tags.append(memento.get('tags', []))
        
        return {
            'texts': torch.stack(texts),
            'images': torch.stack(images),
            'categories': categories,
            'tags': tags
        } 