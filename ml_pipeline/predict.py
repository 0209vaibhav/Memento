import os
import json
import torch
from tqdm import tqdm
import logging
from data_loader import MementoDataLoader
from model import MementoClassifier

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_pipeline/prediction.log'),
        logging.StreamHandler()
    ]
)

def load_categories_and_tags():
    """Load category and tag information."""
    with open('memento_categories.json', 'r') as f:
        categories = json.load(f)
    with open('memento_tags.json', 'r') as f:
        tags = json.load(f)
    return categories, tags

def load_model(model_path: str, num_categories: int, num_tags: int, device: str):
    """Load the trained model."""
    model = MementoClassifier(num_categories, num_tags)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def predict_mementos(model: MementoClassifier, data_loader: MementoDataLoader, 
                    public_mementos: list, categories: list, tags: list,
                    device: str, threshold: float = 0.5):
    """Predict categories and tags for public mementos."""
    predictions = []
    
    for memento in tqdm(public_mementos, desc="Predicting categories and tags"):
        # Process text
        text = f"{memento.get('name', '')} {memento.get('description', '')}"
        text_features = data_loader.process_text(text)
        
        # Process image if available
        if memento.get('media') and len(memento['media']) > 0:
            image_features = data_loader.process_image(memento['media'][0])
        else:
            image_features = torch.zeros((3, 224, 224))
        
        # Add batch dimension
        text_features = text_features.unsqueeze(0)
        image_features = image_features.unsqueeze(0)
        
        # Move to device
        text_features = text_features.to(device)
        image_features = image_features.to(device)
        
        # Get predictions
        with torch.no_grad():
            category_logits, tag_probs = model(image_features, text_features)
            
            # Get category prediction
            category_idx = torch.argmax(category_logits, dim=1).item()
            predicted_category = categories[category_idx]
            
            # Get tag predictions (above threshold)
            tag_indices = (tag_probs > threshold).nonzero().squeeze().tolist()
            if not isinstance(tag_indices, list):
                tag_indices = [tag_indices]
            predicted_tags = [tags[idx] for idx in tag_indices]
            
            # If no tags predicted, use "Other"
            if not predicted_tags:
                predicted_tags = [next(tag for tag in tags if tag["id"] == "other")]
        
        # Create prediction in the exact same structure as scraped mementos
        prediction = {
            "userId": memento.get("userId", "Secret NYC"),
            "location": memento.get("location", None),
            "media": memento.get("media", []),
            "name": memento.get("name", ""),
            "description": memento.get("description", ""),
            "category": f"{predicted_category['symbol']} {predicted_category['name']}",
            "timestamp": memento.get("timestamp", ""),
            "mementoTags": [f"{tag['symbol']} {tag['name']}" for tag in predicted_tags],
            "link": memento.get("link", ""),
            "mementoType": "public"
        }
        
        # Add duration if available
        if "mementoDuration" in memento:
            prediction["mementoDuration"] = memento["mementoDuration"]
        
        # Store prediction confidence in a separate field (not in the original structure)
        prediction["_prediction_confidence"] = {
            "category": torch.softmax(category_logits, dim=1)[0][category_idx].item(),
            "tags": {tag["name"]: tag_probs[0][idx].item() for idx, tag in enumerate(tags)}
        }
        
        predictions.append(prediction)
    
    return predictions

def save_predictions(predictions: list, output_path: str):
    """Save predictions to a JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=4, ensure_ascii=False)

def main():
    # Load configuration
    categories, tags = load_categories_and_tags()
    num_categories = len(categories)
    num_tags = len(tags)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize data loader
    data_loader = MementoDataLoader('firebase-credentials.json')
    
    # Load model
    model = load_model('ml_pipeline/best_model.pth', num_categories, num_tags, device)
    
    # Load public mementos
    logging.info("Loading public mementos...")
    public_mementos = data_loader.load_public_mementos('public-memento-markers')
    
    # Make predictions
    logging.info("Making predictions...")
    predictions = predict_mementos(model, data_loader, public_mementos, categories, tags, device)
    
    # Save predictions
    output_path = 'ml_pipeline/public_memento_predictions.json'
    save_predictions(predictions, output_path)
    logging.info(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    main() 