# FAISS-Based CLIP Fine-tuning Pipeline
# This script combines CLIP fine-tuning with FAISS vector database indexing

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import clip
import numpy as np
import pandas as pd
from PIL import Image
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from collections import defaultdict
import torchvision.transforms as transforms
import faiss
import pickle


# Simple dataset with basic data augmentation
class SimpleNarutoDataset(Dataset):

    def __init__(self, data_dir, split, transform=None, include_unlabeled=False):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.include_unlabeled = include_unlabeled
        self.samples = self._load_samples()
        self.class_to_idx = self._create_class_mapping()
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        print(f"Loaded {len(self.samples)} samples for {split} split")
        print(f"Classes: {list(self.class_to_idx.keys())}")

    def _load_samples(self):
        csv_path = os.path.join(self.data_dir, self.split, '_classes.csv')
        df = pd.read_csv(csv_path)
        samples = []
        class_columns = [col for col in df.columns if col != 'filename']
        for _, row in df.iterrows():
            filename = row['filename']
            image_path = os.path.join(self.data_dir, self.split, filename)
            if not os.path.exists(image_path):
                continue
            active_classes = [col for col in class_columns if row[col] == 1]
            if len(active_classes) == 1:
                label = active_classes[0]
            elif len(active_classes) == 0:
                if self.include_unlabeled:
                    label = "Unlabeled"
                else:
                    continue
            else:
                continue
            samples.append({
                'image_path': image_path,
                'label': label,
                'filename': filename
            })
        return samples

    def _create_class_mapping(self):
        unique_labels = sorted(set(sample['label'] for sample in self.samples))
        return {label: idx for idx, label in enumerate(unique_labels)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label_idx = self.class_to_idx[sample['label']]
        return image, label_idx, sample['label']

    def get_class_distribution(self):
        distribution = defaultdict(int)
        for sample in self.samples:
            distribution[sample['label']] += 1
        return dict(distribution)


# FAISS-based vector database for efficient similarity search
class FAISSVectorDatabase:

    def __init__(self, dimension=512):
        self.dimension = dimension
        self.index = None
        self.data = []

    def create_index(self, index_type="IndexFlatIP"):
        if index_type == "IndexFlatIP":
            self.index = faiss.IndexFlatIP(self.dimension)
        elif index_type == "IndexIVFFlat":
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        print(f"✅ Created FAISS index: {index_type}")

    def add_embeddings(self, embeddings, metadata):
        if self.index is None:
            self.create_index()
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            self.index.train(embeddings)
        self.index.add(embeddings.astype(np.float32))
        self.data.extend(metadata)
        print(f"✅ Added {len(embeddings)} embeddings to FAISS database")

    def search(self, query_embedding, k=5):
        if self.index is None:
            raise ValueError("Index not created")
        query_embedding = query_embedding / np.linalg.norm(query_embedding, keepdims=True)
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(query_embedding, k)
        return scores[0], indices[0]

    def save_to_file(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        index_path = filepath.replace('.pkl', '_faiss.index')
        faiss.write_index(self.index, index_path)
        database_data = {
            'dimension': self.dimension,
            'data': self.data,
            'index_path': index_path
        }
        with open(filepath, 'wb') as f:
            pickle.dump(database_data, f)
        print(f"✅ FAISS database saved to {filepath}")

    def load_from_file(self, filepath):
        with open(filepath, 'rb') as f:
            database_data = pickle.load(f)
        self.dimension = database_data['dimension']
        self.data = database_data['data']
        index_path = database_data['index_path']
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            print(f"Warning: FAISS index file not found: {index_path}")
        print(f"✅ FAISS database loaded from {filepath}")

    def get_stats(self):
        labels = [item['label'] for item in self.data]
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        return {
            'total_entries': len(self.data),
            'unique_labels': len(set(labels)),
            'embedding_dimension': self.dimension,
            'label_distribution': label_counts
        }


# CLIP fine-tuner with FAISS database integration
class FAISSCLIPFineTuner:

    def __init__(self,
                 model_name="ViT-B/32",
                 learning_rate=1e-6,
                 weight_decay=1e-4,
                 batch_size=16,
                 num_epochs=15,
                 use_basic_augmentation=True,
                 device=None):
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.use_basic_augmentation = use_basic_augmentation
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading pre-trained CLIP model: {model_name}")
        self.model, self.clip_preprocess = clip.load(model_name, device=self.device)
        self.model = self.model.float()
        self._create_transforms()
        for param in self.model.transformer.parameters():
            param.requires_grad = False
        total_layers = len(list(self.model.visual.transformer.resblocks))
        layers_to_unfreeze = 2
        for i, layer in enumerate(self.model.visual.transformer.resblocks):
            if i < total_layers - layers_to_unfreeze:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters():
                    param.requires_grad = True
        if hasattr(self.model.visual, 'proj') and self.model.visual.proj is not None:
            self.model.visual.proj.requires_grad = True
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.learning_rates = []
        print(f"✅ FAISS CLIP fine-tuner initialized on {self.device}")
        print(f"Model: {model_name}")
        print(f"Basic augmentation: {'✅' if use_basic_augmentation else '❌'}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

    def _create_transforms(self):
        if self.use_basic_augmentation:
            self.train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomRotation(degrees=5),
                transforms.ColorJitter(brightness=0.05, contrast=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])
            ])
            print("✅ Light data augmentation enabled")
        else:
            self.train_transform = self.clip_preprocess
            print("No augmentation - using standard CLIP preprocessing")
        self.val_transform = self.clip_preprocess

    def create_dataloaders(self, data_dir):
        train_dataset = SimpleNarutoDataset(
            data_dir, 'train', self.train_transform, include_unlabeled=False
        )
        val_dataset = SimpleNarutoDataset(
            data_dir, 'valid', self.val_transform, include_unlabeled=False
        )
        test_dataset = SimpleNarutoDataset(
            data_dir, 'test', self.val_transform, include_unlabeled=False
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=0, pin_memory=True, drop_last=False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=0, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=0, pin_memory=True
        )
        print(f"\nDataset Statistics:")
        print(f"Train: {len(train_dataset)} samples")
        print(f"Validation: {len(val_dataset)} samples")
        print(f"Test: {len(test_dataset)} samples")
        print(f"\nClass distribution (Train):")
        train_dist = train_dataset.get_class_distribution()
        for class_name, count in sorted(train_dist.items(), key=lambda x: x[1], reverse=True):
            print(f"  {class_name}: {count} samples")
        return train_loader, val_loader, test_loader, train_dataset.class_to_idx, train_dataset

    def create_faiss_database(self, dataloader, save_path):
        print("\nCreating FAISS database from training data...")
        self.model.eval()
        all_embeddings = []
        all_metadata = []
        with torch.no_grad():
            for images, labels, label_names in tqdm(dataloader, desc="Processing images"):
                images = images.to(self.device)
                features = self.model.encode_image(images)
                features = features / features.norm(dim=-1, keepdim=True)
                features = features.float().cpu().numpy()
                for i, (feature, label, label_name) in enumerate(zip(features, labels, label_names)):
                    all_embeddings.append(feature)
                    all_metadata.append({
                        'label': label_name,
                        'label_idx': label.item(),
                        'image_path': dataloader.dataset.samples[len(all_embeddings)-1]['image_path'],
                        'filename': dataloader.dataset.samples[len(all_embeddings)-1]['filename']
                    })
        embeddings_matrix = np.array(all_embeddings)
        faiss_db = FAISSVectorDatabase(dimension=embeddings_matrix.shape[1])
        faiss_db.add_embeddings(embeddings_matrix, all_metadata)
        faiss_db.save_to_file(save_path)
        print(f"✅ FAISS database created with {len(all_embeddings)} embeddings")
        return faiss_db

    def compute_class_prototypes(self, dataloader, class_to_idx):
        self.model.eval()
        class_features = {idx: [] for idx in class_to_idx.values()}
        print("Computing class prototypes from training data...")
        with torch.no_grad():
            for images, labels, _ in tqdm(dataloader, desc="Computing prototypes"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                features = self.model.encode_image(images)
                features = features / features.norm(dim=-1, keepdim=True)
                features = features.float()
                for feature, label in zip(features, labels):
                    class_features[label.item()].append(feature.cpu())
        num_classes = len(class_to_idx)
        feature_dim = features.shape[1]
        prototypes = torch.zeros(num_classes, feature_dim)
        for class_idx, feature_list in class_features.items():
            if len(feature_list) > 0:
                prototypes[class_idx] = torch.stack(feature_list).mean(dim=0)
        prototypes = prototypes / prototypes.norm(dim=-1, keepdim=True)
        return prototypes.to(self.device)

    def train_epoch(self, train_loader, prototypes, epoch):
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        for batch_idx, (images, labels, _) in enumerate(progress_bar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            features = self.model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)
            features = features.float()
            logits = torch.matmul(features, prototypes.t()) / 0.07
            loss = self.criterion(logits, labels)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Invalid loss detected, skipping batch {batch_idx}")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
        return total_loss / max(num_batches, 1)

    def validate(self, val_loader, prototypes):
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for images, labels, _ in tqdm(val_loader, desc="Validation"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                features = self.model.encode_image(images)
                features = features / features.norm(dim=-1, keepdim=True)
                features = features.float()
                logits = torch.matmul(features, prototypes.t()) / 0.07
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        accuracy = accuracy_score(all_labels, all_predictions)
        avg_loss = total_loss / len(val_loader)
        return avg_loss, accuracy

    def train(self, data_dir, save_dir="results_faiss_clip_finetuned"):
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "databases"), exist_ok=True)
        train_loader, val_loader, test_loader, class_to_idx, train_dataset = self.create_dataloaders(data_dir)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            eps=1e-8
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.num_epochs, eta_min=self.learning_rate * 0.01
        )
        prototypes = self.compute_class_prototypes(train_loader, class_to_idx)
        best_val_accuracy = 0.0
        patience = 5
        patience_counter = 0
        print(f"\nStarting FAISS-based fine-tuning for {self.num_epochs} epochs...")
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            train_loss = self.train_epoch(train_loader, prototypes, epoch)
            self.train_losses.append(train_loss)
            if epoch % 3 == 0 and epoch > 0:
                prototypes = self.compute_class_prototypes(train_loader, class_to_idx)
            val_loss, val_accuracy = self.validate(val_loader, prototypes)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {val_accuracy:.4f}")
            print(f"Learning Rate: {current_lr:.2e}")
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'prototypes': prototypes,
                    'val_accuracy': val_accuracy,
                    'class_to_idx': class_to_idx,
                    'hyperparameters': {
                        'model_name': self.model_name,
                        'learning_rate': self.learning_rate,
                        'batch_size': self.batch_size,
                        'use_basic_augmentation': self.use_basic_augmentation
                    }
                }
                torch.save(checkpoint, os.path.join(save_dir, 'checkpoints', 'best_model.pth'))
                print(f"Saved best model with validation accuracy: {val_accuracy:.4f}")
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement")
                break
        print(f"\n✅ Training completed!")
        print(f"Best validation accuracy: {best_val_accuracy:.4f}")
        print(f"\nCreating FAISS database with fine-tuned embeddings...")
        faiss_db_path = os.path.join(save_dir, "databases", "naruto_finetuned_faiss_database.pkl")
        faiss_db = self.create_faiss_database(train_loader, faiss_db_path)
        final_checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'prototypes': prototypes,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates,
            'class_to_idx': class_to_idx,
            'faiss_database_path': faiss_db_path
        }
        torch.save(final_checkpoint, os.path.join(save_dir, 'checkpoints', 'final_model.pth'))
        return test_loader, class_to_idx, prototypes, faiss_db

    def evaluate_test_set(self, test_loader, class_to_idx,
                          prototypes=None, model_path=None,
                          save_dir="results_faiss_clip_finetuned"):
        print("\nEVALUATING FAISS CLIP FINE-TUNED MODEL ON TEST SET")
        print("=" * 60)
        if model_path and os.path.exists(model_path):
            print(f"Loading model from: {model_path}")
            try:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                if 'prototypes' in checkpoint:
                    prototypes = checkpoint['prototypes']
                print("✅ Model loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load checkpoint: {e}")
                print("Using current model state for evaluation")
        if prototypes is None:
            print("No prototypes provided, computing from test data")
            prototypes = self.compute_class_prototypes(test_loader, class_to_idx)
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_filenames = []
        all_confidences = []
        per_sample_results = []
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        print(f"Test set size: {len(test_loader.dataset)} samples")
        print(f"Classes: {list(class_to_idx.keys())}")
        print("Evaluating test samples...")
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Test Evaluation")):
                if len(batch_data) == 3:
                    images, labels, filenames = batch_data
                else:
                    images, labels = batch_data
                    filenames = [f"sample_{batch_idx}_{i}" for i in range(len(images))]
                images = images.to(self.device)
                labels = labels.to(self.device)
                features = self.model.encode_image(images)
                features = features / features.norm(dim=-1, keepdim=True)
                features = features.float()
                logits = torch.matmul(features, prototypes.t()) / 0.07
                confidences = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                batch_predictions = predictions.cpu().numpy()
                batch_labels = labels.cpu().numpy()
                batch_confidences = confidences.cpu().numpy()
                all_predictions.extend(batch_predictions)
                all_labels.extend(batch_labels)
                all_filenames.extend(filenames)
                all_confidences.extend(batch_confidences)
                for i in range(len(batch_predictions)):
                    pred_class = idx_to_class[batch_predictions[i]]
                    true_class = idx_to_class[batch_labels[i]]
                    confidence = batch_confidences[i][batch_predictions[i]]
                    per_sample_results.append({
                        'filename': filenames[i],
                        'true_class': true_class,
                        'predicted_class': pred_class,
                        'confidence': float(confidence),
                        'correct': pred_class == true_class,
                        'top1_prediction': pred_class
                    })
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_confidences = np.array(all_confidences)
        overall_accuracy = accuracy_score(all_labels, all_predictions)
        correct_predictions = np.sum(all_predictions == all_labels)
        total_predictions = len(all_labels)
        print(f"\nTEST SET EVALUATION RESULTS")
        print("=" * 50)
        print(f"Overall Accuracy: {overall_accuracy:.4f} ({correct_predictions}/{total_predictions})")
        print(f"Percentage of objects correctly identified: {overall_accuracy * 100:.2f}%")
        class_report = classification_report(all_labels, all_predictions,
                                             target_names=list(class_to_idx.keys()),
                                             output_dict=True, zero_division=0)
        print(f"\nPER-CLASS PERFORMANCE:")
        print("-" * 40)
        per_class_metrics = {}
        for class_name, idx in class_to_idx.items():
            class_mask = all_labels == idx
            if np.any(class_mask):
                class_predictions = all_predictions[class_mask]
                class_accuracy = np.mean(class_predictions == idx)
                class_count = np.sum(class_mask)
                correct_count = np.sum(class_predictions == idx)
                per_class_metrics[class_name] = {
                    'accuracy': float(class_accuracy),
                    'correct': int(correct_count),
                    'total': int(class_count),
                    'precision': float(class_report[class_name]['precision']),
                    'recall': float(class_report[class_name]['recall']),
                    'f1_score': float(class_report[class_name]['f1-score'])
                }
                print(f"  {class_name:>8}: {class_accuracy:.4f} ({correct_count}/{class_count}) "
                      f"P:{class_report[class_name]['precision']:.3f} "
                      f"R:{class_report[class_name]['recall']:.3f} "
                      f"F1:{class_report[class_name]['f1-score']:.3f}")
        os.makedirs(os.path.join(save_dir, "evaluation"), exist_ok=True)
        evaluation_results = {
            'overall_metrics': {
                'accuracy': float(overall_accuracy),
                'correct_predictions': int(correct_predictions),
                'total_predictions': int(total_predictions),
                'percentage_correct': float(overall_accuracy * 100)
            },
            'per_class_metrics': per_class_metrics,
            'class_to_idx': class_to_idx,
            'classification_report': class_report,
            'model_info': {
                'model_name': self.model_name,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'num_epochs': self.num_epochs,
                'use_basic_augmentation': self.use_basic_augmentation,
                'fine_tuning_type': 'faiss_clip'
            }
        }
        report_path = os.path.join(save_dir, "evaluation", "test_evaluation_report.json")
        with open(report_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        print(f"Evaluation report saved to: {report_path}")
        per_sample_path = os.path.join(save_dir, "evaluation", "per_sample_predictions.json")
        with open(per_sample_path, 'w') as f:
            json.dump(per_sample_results, f, indent=2)
        print(f"Per-sample predictions saved to: {per_sample_path}")
        return evaluation_results

    def plot_training_curves(self, save_path=None):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax2.plot(epochs, self.val_accuracies, 'g-', label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax3.plot(epochs, self.learning_rates, 'orange', label='Learning Rate', linewidth=2)
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax4.scatter(self.val_losses, self.val_accuracies, c=epochs, cmap='viridis', s=50, alpha=0.7)
        ax4.set_xlabel('Validation Loss')
        ax4.set_ylabel('Validation Accuracy')
        ax4.set_title('Loss vs Accuracy Progression')
        ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to: {save_path}")
        plt.show()


if __name__ == "__main__":
    # FAISS-BASED CLIP FINE-TUNING CONFIGURATION
    print("FAISS-BASED CLIP FINE-TUNING")
    print("=" * 50)
    print("Training CLIP with FAISS database integration")
    config = {
        'model_name': 'ViT-B/32',
        'learning_rate': 1e-6,
        'weight_decay': 1e-4,
        'batch_size': 16,
        'num_epochs': 15,
        'use_basic_augmentation': True,
    }
    data_dir = r"./Anime -Naruto-.v1i.multiclass"
    save_dir = "results_faiss_clip_finetuned"
    print(f"\nCONFIGURATION:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    fine_tuner = FAISSCLIPFineTuner(**config)
    print(f"\nStarting FAISS-based fine-tuning...")
    test_loader, class_to_idx, prototypes, faiss_db = fine_tuner.train(
        data_dir,
        save_dir=save_dir
    )
    plot_path = os.path.join(save_dir, "plots", "faiss_training_curves.png")
    fine_tuner.plot_training_curves(plot_path)
    print(f"\n✅ FAISS-based fine-tuning completed!")
    print(f"Results saved in '{save_dir}/' directory")
    print("\n" + "="*60)
    print("STARTING TEST SET EVALUATION")
    print("="*60)
    best_model_path = os.path.join(save_dir, "checkpoints", "best_model.pth")
    evaluation_results = fine_tuner.evaluate_test_set(
        test_loader=test_loader,
        class_to_idx=class_to_idx,
        prototypes=prototypes,
        model_path=best_model_path if os.path.exists(best_model_path) else None,
        save_dir=save_dir
    )
    print("\n" + "="*60)
    print("FAISS CLIP FINE-TUNING EVALUATION COMPLETE")
    print("="*60)
    overall_metrics = evaluation_results['overall_metrics']
    print(f"TEST ACCURACY: {overall_metrics['accuracy']:.4f}")
    print(f"PERCENTAGE CORRECT: {overall_metrics['percentage_correct']:.2f}%")
    print(f"✅ CORRECT PREDICTIONS: {overall_metrics['correct_predictions']}")
    print(f"TOTAL TEST SAMPLES: {overall_metrics['total_predictions']}")
    print(f"\nPER-CLASS ACCURACY BREAKDOWN:")
    for class_name, metrics in evaluation_results['per_class_metrics'].items():
        accuracy_percent = metrics['accuracy'] * 100
        print(f"   {class_name:>8}: {accuracy_percent:>6.2f}% ({metrics['correct']}/{metrics['total']})")
    print(f"\nFAISS INTEGRATION FEATURES:")
    print(f"   ✅ FAISS vector database created with fine-tuned embeddings")
    print(f"   ✅ Efficient similarity search with cosine similarity")
    print(f"   ✅ Prototype-based contrastive learning")
    print(f"   ✅ Database saved to: {save_dir}/databases/")
    print(f"\nFAISS CLIP FINE-TUNING COMPLETE!")
    print(f"The fine-tuned model and FAISS database are ready for deployment!")

