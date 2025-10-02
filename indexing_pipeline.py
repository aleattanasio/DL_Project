import os
import torch
import clip
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm


class ImageLoader:
    # Load images from dataset directory using CSV labels

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def load_images_from_csv(self, split='train'):
        split_path = os.path.join(self.dataset_path, split)
        csv_path = os.path.join(split_path, '_classes.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        df = pd.read_csv(csv_path)
        class_names = [col for col in df.columns if col != 'filename']
        images_data = []
        for _, row in df.iterrows():
            filename = row['filename']
            image_path = os.path.join(split_path, filename)
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                continue
            active_classes = [class_name for class_name in class_names if row[class_name] == 1]
            if len(active_classes) == 1:
                label = active_classes[0]
            elif len(active_classes) == 0:
                label = "Unlabeled"
            else:
                label = "_".join(active_classes)
            images_data.append({
                'image_path': image_path,
                'label': label,
                'filename': filename,
                'split': split
            })
        print(f"Loaded {len(images_data)} images from {split} split")
        return images_data


class EmbeddingModel:
    # Pre-trained or externally provided (fine-tuned) CLIP model for image embeddings

    def __init__(self, model_name="ViT-B/32", model=None, preprocess=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if model is not None:
            self.model = model
            if preprocess is None:
                print("Missing preprocess: loading standard preprocess for", model_name)
                _tmp_model, self.preprocess = clip.load(model_name, device=self.device)
                del _tmp_model
            else:
                self.preprocess = preprocess
            print("EmbeddingModel: using external (fine-tuned) CLIP model")
        else:
            print(f"Loading CLIP model {model_name} on {self.device}")
            self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

    def encode_image(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            return image_features.cpu().numpy().flatten()
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None

    def encode_images_batch(self, image_paths, batch_size=32):
        embeddings = []
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images"):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            valid_indices = []
            for j, path in enumerate(batch_paths):
                try:
                    image = Image.open(path).convert('RGB')
                    image_input = self.preprocess(image)
                    batch_images.append(image_input)
                    valid_indices.append(i + j)
                except Exception as e:
                    print(f"Error loading image {path}: {e}")
                    embeddings.append(None)
            if batch_images:
                batch_tensor = torch.stack(batch_images).to(self.device)
                with torch.no_grad():
                    batch_features = self.model.encode_image(batch_tensor)
                    batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
                    batch_features = batch_features.cpu().numpy()
                for idx, features in zip(valid_indices, batch_features):
                    embeddings.append(features)
        return embeddings


class VectorDatabase:
    # In-memory vector database for storing embeddings and metadata

    def __init__(self):
        self.data = []
        self.index_to_id = {}
        self.next_id = 0

    def add_entry(self, image_path, label, embedding, metadata=None):
        if embedding is None:
            return None
        entry_id = self.next_id
        self.next_id += 1
        entry = {
            'id': entry_id,
            'image_path': image_path,
            'label': label,
            'embedding': embedding,
            'metadata': metadata or {}
        }
        self.data.append(entry)
        self.index_to_id[len(self.data) - 1] = entry_id
        return entry_id

    def get_entry(self, entry_id):
        for entry in self.data:
            if entry['id'] == entry_id:
                return entry
        return None

    def get_all_entries(self):
        return self.data.copy()

    def get_embeddings_matrix(self):
        if not self.data:
            return np.array([])
        return np.vstack([entry['embedding'] for entry in self.data])

    def get_labels(self):
        return [entry['label'] for entry in self.data]

    def get_stats(self):
        if not self.data:
            return {'total_entries': 0, 'labels': {}}
        labels = self.get_labels()
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        return {
            'total_entries': len(self.data),
            'unique_labels': len(label_counts),
            'label_distribution': label_counts,
            'embedding_dimension': self.data[0]['embedding'].shape[0] if self.data else 0
        }

    def save_to_file(self, filepath):
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'data': self.data,
                'next_id': self.next_id
            }, f)
        print(f"Database saved to {filepath}")

    def load_from_file(self, filepath):
        import pickle
        with open(filepath, 'rb') as f:
            saved_data = pickle.load(f)
            self.data = saved_data['data']
            self.next_id = saved_data['next_id']
            self.index_to_id = {i: entry['id'] for i, entry in enumerate(self.data)}
        print(f"Database loaded from {filepath}")


class IndexingPipeline:
    # Pipeline orchestrating image loading, embedding, and storage

    def __init__(self, dataset_path, model_name="ViT-B/32"):
        self.dataset_path = dataset_path
        self.image_loader = ImageLoader(dataset_path)
        self.embedding_model = EmbeddingModel(model_name)
        self.vector_db = VectorDatabase()

    def process_dataset(self, splits=['train', 'valid', 'test'], batch_size=32):
        print("Starting indexing pipeline...")
        for split in splits:
            print(f"\nProcessing {split} split...")
            try:
                images_data = self.image_loader.load_images_from_csv(split)
                if not images_data:
                    print(f"No images found in {split} split")
                    continue
                image_paths = [item['image_path'] for item in images_data]
                embeddings = self.embedding_model.encode_images_batch(image_paths, batch_size)
                for i, (image_data, embedding) in enumerate(zip(images_data, embeddings)):
                    if embedding is not None:
                        metadata = {
                            'filename': image_data['filename'],
                            'split': image_data['split']
                        }
                        self.vector_db.add_entry(
                            image_path=image_data['image_path'],
                            label=image_data['label'],
                            embedding=embedding,
                            metadata=metadata
                        )
            except Exception as e:
                print(f"Error processing {split} split: {e}")
                continue
        stats = self.vector_db.get_stats()
        print(f"\n=== Indexing Complete ===")
        print(f"Total entries: {stats['total_entries']}")
        print(f"Unique labels: {stats['unique_labels']}")
        print(f"Embedding dimension: {stats['embedding_dimension']}")
        print(f"Label distribution: {stats['label_distribution']}")
        return self.vector_db


if __name__ == "__main__":
    dataset_path = r"./Anime -Naruto-.v1i.multiclass"
    pipeline = IndexingPipeline(dataset_path)
    vector_db = pipeline.process_dataset()
    vector_db.save_to_file("naruto_embeddings.pkl")
