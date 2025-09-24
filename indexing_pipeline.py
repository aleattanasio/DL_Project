import os
import torch
import clip
from PIL import Image
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm


class ImageLoader:
    """Loads images from dataset directory using folder names as ground-truth labels."""

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    def load_images_from_csv(self, split: str = 'train') -> List[Dict]:
        """
        Load images and labels from CSV files in the dataset.

        Args:
            split: Dataset split ('train', 'valid', 'test')

        Returns:
            List of dictionaries containing image paths and labels
        """
        split_path = os.path.join(self.dataset_path, split)
        csv_path = os.path.join(split_path, '_classes.csv')

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Load CSV file
        df = pd.read_csv(csv_path)

        # Class names from CSV columns (excluding filename)
        class_names = [col for col in df.columns if col != 'filename']

        images_data = []

        for _, row in df.iterrows():
            filename = row['filename']
            image_path = os.path.join(split_path, filename)

            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                continue

            # Find the active class (where value is 1)
            active_classes = [class_name for class_name in class_names if row[class_name] == 1]

            # Handle case where no class is active or multiple classes are active
            if len(active_classes) == 1:
                label = active_classes[0]
            elif len(active_classes) == 0:
                label = "Unlabeled"
            else:
                # If multiple classes are active, join them
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
    """Pre-trained CLIP model for converting images to high-dimensional feature vectors."""

    def __init__(self, model_name: str = "ViT-B/32", device: str = None):
        """
        Initialize CLIP model.

        Args:
            model_name: CLIP model variant to use
            device: Device to run model on (auto-detected if None)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading CLIP model {model_name} on {self.device}")

        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

    def encode_image(self, image_path: str) -> np.ndarray:
        """
        Convert single image to embedding vector.

        Args:
            image_path: Path to image file

        Returns:
            Normalized embedding vector as numpy array
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)

            # Generate embedding
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                # Normalize the features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            return image_features.cpu().numpy().flatten()

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None

    def encode_images_batch(self, image_paths: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """
        Convert multiple images to embedding vectors in batches.

        Args:
            image_paths: List of image file paths
            batch_size: Number of images to process at once

        Returns:
            List of embedding vectors
        """
        embeddings = []

        for i in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images"):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            valid_indices = []

            # Load and preprocess batch
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
                # Process batch
                batch_tensor = torch.stack(batch_images).to(self.device)

                with torch.no_grad():
                    batch_features = self.model.encode_image(batch_tensor)
                    batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
                    batch_features = batch_features.cpu().numpy()

                # Add batch results to embeddings list
                for idx, features in zip(valid_indices, batch_features):
                    embeddings.append(features)

        return embeddings


class VectorDatabase:
    """Simple in-memory vector database for storing image embeddings and metadata."""

    def __init__(self):
        """Initialize empty vector database."""
        self.data = []
        self.index_to_id = {}
        self.next_id = 0

    def add_entry(self, image_path: str, label: str, embedding: np.ndarray, metadata: Dict = None) -> int:
        """
        Add an entry to the vector database.

        Args:
            image_path: Path to the image file
            label: Character label
            embedding: Feature vector
            metadata: Additional metadata

        Returns:
            Entry ID
        """
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

    def get_entry(self, entry_id: int) -> Dict:
        """Get entry by ID."""
        for entry in self.data:
            if entry['id'] == entry_id:
                return entry
        return None

    def get_all_entries(self) -> List[Dict]:
        """Get all entries in the database."""
        return self.data.copy()

    def get_embeddings_matrix(self) -> np.ndarray:
        """Get all embeddings as a matrix."""
        if not self.data:
            return np.array([])
        return np.vstack([entry['embedding'] for entry in self.data])

    def get_labels(self) -> List[str]:
        """Get all labels."""
        return [entry['label'] for entry in self.data]

    def get_stats(self) -> Dict:
        """Get database statistics."""
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

    def save_to_file(self, filepath: str):
        """Save database to file."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'data': self.data,
                'next_id': self.next_id
            }, f)
        print(f"Database saved to {filepath}")

    def load_from_file(self, filepath: str):
        """Load database from file."""
        import pickle
        with open(filepath, 'rb') as f:
            saved_data = pickle.load(f)
            self.data = saved_data['data']
            self.next_id = saved_data['next_id']
            # Rebuild index
            self.index_to_id = {i: entry['id'] for i, entry in enumerate(self.data)}
        print(f"Database loaded from {filepath}")


class IndexingPipeline:
    """Main indexing pipeline that orchestrates the image loading, embedding, and storage."""

    def __init__(self, dataset_path: str, model_name: str = "ViT-B/32"):
        """
        Initialize the indexing pipeline.

        Args:
            dataset_path: Path to the dataset directory
            model_name: CLIP model variant to use
        """
        self.dataset_path = dataset_path
        self.image_loader = ImageLoader(dataset_path)
        self.embedding_model = EmbeddingModel(model_name)
        self.vector_db = VectorDatabase()

    def process_dataset(self, splits: List[str] = ['train', 'valid', 'test'], batch_size: int = 32) -> VectorDatabase:
        """
        Process the entire dataset and build the vector database.

        Args:
            splits: Dataset splits to process
            batch_size: Batch size for embedding generation

        Returns:
            Populated vector database
        """
        print("Starting indexing pipeline...")

        for split in splits:
            print(f"\nProcessing {split} split...")

            try:
                # Load images and labels
                images_data = self.image_loader.load_images_from_csv(split)

                if not images_data:
                    print(f"No images found in {split} split")
                    continue

                # Extract image paths for batch processing
                image_paths = [item['image_path'] for item in images_data]

                # Generate embeddings
                embeddings = self.embedding_model.encode_images_batch(image_paths, batch_size)

                # Add to vector database
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

        # Print final statistics
        stats = self.vector_db.get_stats()
        print(f"\n=== Indexing Complete ===")
        print(f"Total entries: {stats['total_entries']}")
        print(f"Unique labels: {stats['unique_labels']}")
        print(f"Embedding dimension: {stats['embedding_dimension']}")
        print(f"Label distribution: {stats['label_distribution']}")

        return self.vector_db


if __name__ == "__main__":
    # Example usage
    dataset_path = r"C:\Users\Utente\PycharmProjects\DL_Exam_v2\Anime -Naruto-.v1i.multiclass"

    # Initialize and run pipeline
    pipeline = IndexingPipeline(dataset_path)
    vector_db = pipeline.process_dataset()

    # Save the database
    vector_db.save_to_file("naruto_embeddings.pkl")
