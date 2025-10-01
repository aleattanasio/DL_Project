# BLIP-2 fine-tuning pipeline for Naruto character recognition

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import Blip2Processor, Blip2Model
import numpy as np
import pandas as pd
from PIL import Image
import os
import json
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import logging
from datetime import datetime
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# BLIP-2 model wrapper for generating image and text embeddings
class BLIP2EmbeddingModel:

    def __init__(self, model_name="Salesforce/blip2-opt-2.7b", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        logger.info(f"Loading BLIP-2 model: {model_name}")
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2Model.from_pretrained(model_name, torch_dtype=torch.float16)
        self.model.to(self.device)
        self.model.eval()
        self.embedding_dim = self.model.config.qformer_config.hidden_size
        logger.info(f"BLIP-2 model loaded successfully on {self.device}")
        logger.info(f"Embedding dimension: {self.embedding_dim}")

    def encode_image(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.get_qformer_features(**inputs)
                image_embeds = outputs.last_hidden_state
                pooled_embeds = image_embeds.mean(dim=1)
                pooled_embeds = pooled_embeds / pooled_embeds.norm(dim=-1, keepdim=True)
            return pooled_embeds.cpu().numpy().flatten()
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            return None

    def encode_text(self, text):
        try:
            inputs = self.processor(text=text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.get_qformer_features(**inputs)
                text_embeds = outputs.last_hidden_state
                pooled_embeds = text_embeds.mean(dim=1)
                pooled_embeds = pooled_embeds / pooled_embeds.norm(dim=-1, keepdim=True)
            return pooled_embeds.cpu().numpy().flatten()
        except Exception as e:
            logger.error(f"Error encoding text '{text}': {e}")
            return None


# Dataset class for Naruto character images with BLIP-2 processing
class NarutoDatasetBLIP2(Dataset):

    def __init__(self, data_dir, split='train', processor=None):
        self.data_dir = data_dir
        self.split = split
        self.processor = processor
        self.data = self._load_data()
        self.class_names = sorted(list(set([item['label'] for item in self.data])))
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        logger.info(f"Loaded {len(self.data)} samples for {split} split")
        logger.info(f"Classes: {self.class_names}")

    def _load_data(self):
        data = []
        split_dir = os.path.join(self.data_dir, self.split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        classes_file = os.path.join(split_dir, '_classes.csv')
        if os.path.exists(classes_file):
            classes_df = pd.read_csv(classes_file)
            possible_class_columns = ['Gara', 'Naruto', 'Sakura', 'Tsunade', 'Unlabeled']
            if all(col in classes_df.columns for col in possible_class_columns):
                filename_to_class = {}
                for _, row in classes_df.iterrows():
                    filename = row['filename']
                    for class_name in possible_class_columns:
                        if row[class_name] == 1:
                            filename_to_class[filename] = class_name
                            break
                    else:
                        filename_to_class[filename] = 'Unknown'
            elif 'class' in classes_df.columns:
                filename_to_class = dict(zip(classes_df['filename'], classes_df['class']))
            else:
                logger.warning(f"Unexpected CSV format in {classes_file}. Attempting to infer from filenames.")
                filename_to_class = {}
                for file in os.listdir(split_dir):
                    if file.endswith(('.jpg', '.jpeg', '.png')):
                        class_name = file.split('_')[0] if '_' in file else 'unknown'
                        filename_to_class[file] = class_name
        else:
            filename_to_class = {}
            for file in os.listdir(split_dir):
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    class_name = file.split('_')[0] if '_' in file else 'unknown'
                    filename_to_class[file] = class_name
        for filename, class_name in filename_to_class.items():
            image_path = os.path.join(split_dir, filename)
            if os.path.exists(image_path):
                data.append({
                    'image_path': image_path,
                    'label': class_name,
                    'filename': filename
                })
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            image = Image.open(item['image_path']).convert('RGB')
        except Exception as e:
            logger.warning(f"Error loading image {item['image_path']}: {e}")
            image = Image.new('RGB', (224, 224), color='gray')
        class_idx = self.class_to_idx[item['label']]
        if self.processor:
            processed = self.processor(images=image, return_tensors="pt")
            for key in processed:
                if torch.is_tensor(processed[key]):
                    processed[key] = processed[key].squeeze(0)
            return processed, torch.tensor(class_idx, dtype=torch.long), item
        else:
            return image, torch.tensor(class_idx, dtype=torch.long), item


# Fine-tuning pipeline for BLIP-2 on Naruto character recognition
class BLIP2FineTuner:

    def __init__(self,
                 model_name="Salesforce/blip2-opt-2.7b",
                 data_dir="Anime -Naruto-.v1i.multiclass",
                 output_dir="results_blip2_finetuned",
                 device=None):
        self.model_name = model_name
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'reports'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'databases'), exist_ok=True)
        self.processor = Blip2Processor.from_pretrained(model_name, use_fast=True)
        self.model = None
        self.embedding_model = None
        self.train_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epochs': []
        }
        self.datasets = {}
        self.class_names = []
        self.num_classes = 0

    def prepare_datasets(self):
        logger.info("Preparing datasets...")
        self.datasets['train'] = NarutoDatasetBLIP2(self.data_dir, 'train', self.processor)
        self.datasets['valid'] = NarutoDatasetBLIP2(self.data_dir, 'valid', self.processor)
        self.datasets['test'] = NarutoDatasetBLIP2(self.data_dir, 'test', self.processor)
        self.class_names = self.datasets['train'].class_names
        self.num_classes = len(self.class_names)
        logger.info(f"Dataset prepared: {self.num_classes} classes")
        logger.info(f"Train: {len(self.datasets['train'])}, "
                    f"Valid: {len(self.datasets['valid'])}, "
                    f"Test: {len(self.datasets['test'])}")

    def create_model(self):
        logger.info("Creating BLIP-2 model with classification head...")
        base_model = Blip2Model.from_pretrained(self.model_name, torch_dtype=torch.float16)

        class BLIP2Classifier(nn.Module):
            def __init__(self, base_model, num_classes, embedding_dim):
                super().__init__()
                self.base_model = base_model
                self.classifier = nn.Sequential(
                    nn.Linear(embedding_dim, 512),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, num_classes)
                )
                for param in self.base_model.parameters():
                    param.requires_grad = False

            def forward(self, pixel_values):
                with torch.no_grad():
                    outputs = self.base_model.get_qformer_features(pixel_values=pixel_values)
                    features = outputs.last_hidden_state.mean(dim=1)
                logits = self.classifier(features.float())
                return logits

        embedding_dim = base_model.config.qformer_config.hidden_size
        self.model = BLIP2Classifier(base_model, self.num_classes, embedding_dim)
        self.model.to(self.device)
        logger.info(f"Model created with {self.num_classes} output classes")
        return self.model

    def train_epoch(self, dataloader, optimizer, criterion, epoch):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch_idx, (inputs, targets, metadata) in enumerate(progress_bar):
            pixel_values = inputs['pixel_values'].to(self.device)
            targets = targets.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(pixel_values)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            avg_loss = total_loss / (batch_idx + 1)
            acc = 100. * correct / total
            progress_bar.set_postfix({'Loss': f'{avg_loss:.4f}', 'Acc': f'{acc:.2f}%'})
        epoch_loss = total_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc

    def validate_epoch(self, dataloader, criterion):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets, metadata in tqdm(dataloader, desc="Validating"):
                pixel_values = inputs['pixel_values'].to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(pixel_values)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        epoch_loss = total_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc

    def train(self,
              num_epochs=20,
              batch_size=16,
              learning_rate=1e-4,
              weight_decay=1e-4,
              patience=3):
        logger.info("Starting BLIP-2 fine-tuning...")
        self.prepare_datasets()
        train_loader = DataLoader(self.datasets['train'], batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(self.datasets['valid'], batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        self.create_model()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW([p for p in self.model.parameters() if p.requires_grad],
                                lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)
        best_val_acc = 0.0
        epochs_without_improvement = 0
        start_time = datetime.now()
        logger.info(f"Training configuration:")
        logger.info(f"  Epochs: {num_epochs}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Early stopping patience: {patience} epochs")
        logger.info(f"  Device: {self.device}")
        for epoch in range(1, num_epochs + 1):
            logger.info(f"\nEpoch {epoch}/{num_epochs}")
            logger.info("-" * 40)
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion, epoch)
            val_loss, val_acc = self.validate_epoch(val_loader, criterion)
            scheduler.step()
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_acc'].append(val_acc)
            self.train_history['epochs'].append(epoch)
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_without_improvement = 0
                self.save_model('best_model.pth', epoch, val_acc)
                logger.info(f"New best validation accuracy: {val_acc:.2f}% - Model saved!")
            else:
                epochs_without_improvement += 1
                logger.info(f"No improvement for {epochs_without_improvement} epoch(s)")
            if epochs_without_improvement >= patience:
                logger.info(f"\nEarly stopping triggered!")
                logger.info(f"No improvement in validation accuracy for {patience} consecutive epochs")
                logger.info(f"Best validation accuracy achieved: {best_val_acc:.2f}%")
                break
        self.save_model('final_model.pth', epoch, val_acc)
        training_time = datetime.now() - start_time
        logger.info(f"\nTraining completed in {training_time}")
        if epochs_without_improvement >= patience:
            logger.info(f"Training stopped early at epoch {epoch}")
        else:
            logger.info(f"Training completed all {num_epochs} epochs")
        logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
        self.save_training_history()
        self.plot_training_curves()
        return best_val_acc

    def save_model(self, filename, epoch, val_acc):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_acc': val_acc,
            'class_names': self.class_names,
            'num_classes': self.num_classes,
            'model_name': self.model_name,
            'train_history': self.train_history
        }
        save_path = os.path.join(self.output_dir, 'checkpoints', filename)
        torch.save(checkpoint, save_path)
        logger.info(f"Model saved: {save_path}")

    def save_training_history(self):
        history_path = os.path.join(self.output_dir, 'reports', 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        logger.info(f"Training history saved: {history_path}")

    def plot_training_curves(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        epochs = self.train_history['epochs']
        ax1.plot(epochs, self.train_history['train_loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, self.train_history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('BLIP-2 Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        ax2.plot(epochs, self.train_history['train_acc'], 'b-', label='Training Accuracy')
        ax2.plot(epochs, self.train_history['val_acc'], 'r-', label='Validation Accuracy')
        ax2.set_title('BLIP-2 Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'plots', 'blip2_training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Training curves saved: {plot_path}")

    def evaluate(self, split='test'):
        logger.info(f"Evaluating on {split} set...")
        best_model_path = os.path.join(self.output_dir, 'checkpoints', 'best_model.pth')
        if not os.path.exists(best_model_path):
            logger.error("No trained model found. Please train the model first.")
            return
        checkpoint = torch.load(best_model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        dataset = self.datasets[split]
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
        all_predictions = []
        all_targets = []
        all_metadata = []
        with torch.no_grad():
            for inputs, targets, metadata in tqdm(dataloader, desc=f"Evaluating {split}"):
                pixel_values = inputs['pixel_values'].to(self.device)
                outputs = self.model(pixel_values)
                _, predicted = outputs.max(1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.numpy())
                if isinstance(metadata, (list, tuple)):
                    all_metadata.extend(metadata)
                else:
                    all_metadata.append(metadata)
        accuracy = accuracy_score(all_targets, all_predictions)
        class_report = classification_report(all_targets, all_predictions,
                                             target_names=self.class_names,
                                             output_dict=True)
        cm = confusion_matrix(all_targets, all_predictions)
        results = {
            'accuracy': float(accuracy),
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'class_names': self.class_names,
            'num_samples': len(all_targets),
            'model_name': self.model_name,
            'evaluation_split': split
        }
        detailed_results = []
        for i, (pred, target, meta) in enumerate(zip(all_predictions, all_targets, all_metadata)):
            detailed_results.append({
                'filename': meta['filename'],
                'true_label': self.class_names[target],
                'predicted_label': self.class_names[pred],
                'correct': bool(pred == target),
                'image_path': meta['image_path']
            })
        results['detailed_predictions'] = detailed_results
        report_path = os.path.join(self.output_dir, 'reports', f'{split}_evaluation_report.json')
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        self.plot_confusion_matrix(cm, split)
        logger.info(f"Evaluation completed:")
        logger.info(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        logger.info(f"  Results saved: {report_path}")
        return results

    def plot_confusion_matrix(self, cm, split):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title(f'BLIP-2 Confusion Matrix - {split.title()} Set')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'plots', f'confusion_matrix_{split}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Confusion matrix saved: {plot_path}")

    def create_embeddings_database(self):
        logger.info("Creating BLIP-2 embeddings database using fine-tuned model...")
        best_model_path = os.path.join(self.output_dir, 'checkpoints', 'best_model.pth')
        if not os.path.exists(best_model_path):
            logger.error("No fine-tuned model found. Cannot create embeddings database.")
            return None
        checkpoint = torch.load(best_model_path, map_location=self.device)
        if self.model is None:
            self.create_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        logger.info("Using fine-tuned BLIP-2 model for embeddings extraction")
        database_data = []
        train_dataset = self.datasets['train']
        for item in tqdm(train_dataset.data, desc="Creating fine-tuned embeddings"):
            try:
                image = Image.open(item['image_path']).convert('RGB')
                inputs = self.processor(images=image, return_tensors="pt")
                pixel_values = inputs['pixel_values'].to(self.device)
                with torch.no_grad():
                    outputs = self.model.base_model.get_qformer_features(pixel_values=pixel_values)
                    features = outputs.last_hidden_state.mean(dim=1)
                    features = features / features.norm(dim=-1, keepdim=True)
                    embedding = features.cpu().numpy().flatten()
                database_data.append({
                    'image_path': item['image_path'],
                    'label': item['label'],
                    'embedding': embedding,
                    'filename': item['filename']
                })
            except Exception as e:
                logger.warning(f"Error processing image {item['image_path']}: {e}")
                continue
        if not database_data:
            logger.error("No embeddings were created successfully.")
            return None
        embeddings_matrix = np.array([item['embedding'] for item in database_data])
        embedding_dim = self.model.base_model.config.qformer_config.hidden_size
        database = {
            'data': database_data,
            'embeddings_matrix': embeddings_matrix,
            'class_names': self.class_names,
            'model_info': {
                'model_name': self.model_name,
                'embedding_dim': embedding_dim,
                'fine_tuned': True,
                'fine_tuned_checkpoint': best_model_path,
                'creation_date': datetime.now().isoformat()
            }
        }
        db_path = os.path.join(self.output_dir, 'databases', 'naruto_blip2_finetuned_embeddings.pkl')
        with open(db_path, 'wb') as f:
            pickle.dump(database, f)
        logger.info(f"Fine-tuned embeddings database created: {db_path}")
        logger.info(f"Database contains {len(database_data)} embeddings")
        logger.info(f"Embedding dimension: {embedding_dim}")
        logger.info("✅ Using fine-tuned model features for better character-specific representations!")
        return database


def main():
    parser = argparse.ArgumentParser(description='BLIP-2 Fine-tuning for Naruto Character Recognition')
    parser.add_argument('--data_dir', type=str, default='Anime -Naruto-.v1i.multiclass',
                        help='Directory containing the dataset')
    parser.add_argument('--output_dir', type=str, default='results_blip2_finetuned',
                        help='Directory to save results')
    parser.add_argument('--model_name', type=str, default='Salesforce/blip2-opt-2.7b',
                        help='BLIP-2 model variant to use')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for regularization')
    parser.add_argument('--evaluate_only', action='store_true',
                        help='Only run evaluation (skip training)')
    args = parser.parse_args()
    fine_tuner = BLIP2FineTuner(
        model_name=args.model_name,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    try:
        if not args.evaluate_only:
            logger.info("Starting BLIP-2 fine-tuning pipeline...")
            best_val_acc = fine_tuner.train(
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay
            )
            logger.info(f"Training completed with best validation accuracy: {best_val_acc:.2f}%")
        else:
            fine_tuner.prepare_datasets()
            fine_tuner.create_model()
        logger.info("Evaluating trained model...")
        test_results = fine_tuner.evaluate('test')
        logger.info("Creating embeddings database...")
        database = fine_tuner.create_embeddings_database()
        logger.info("BLIP-2 fine-tuning pipeline completed successfully!")
        logger.info(f"Final test accuracy: {test_results['accuracy']:.4f}")
    except Exception as e:
        logger.error(f"Error in fine-tuning pipeline: {e}")
        raise


if __name__ == "__main__":
    main()
