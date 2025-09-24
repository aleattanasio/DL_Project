import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from typing import List, Dict, Tuple, Optional
import clip
from indexing_pipeline import VectorDatabase, EmbeddingModel
import os
from urllib.request import urlretrieve


class SegmentationModel:
    """Pre-trained SAM model for object segmentation."""

    def __init__(self, model_type: str = "vit_h", device: str = None):
        """
        Initialize SAM model.

        Args:
            model_type: SAM model variant ('vit_h', 'vit_l', 'vit_b')
            device: Device to run model on (auto-detected if None)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type

        # Download model checkpoint if not exists
        self.checkpoint_path = self._download_sam_checkpoint(model_type)

        print(f"Loading SAM model {model_type} on {self.device}")

        # Initialize SAM model
        self.sam = sam_model_registry[model_type](checkpoint=self.checkpoint_path)
        self.sam.to(device=self.device)

        # Initialize mask generator for automatic segmentation
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
        )

        # Initialize predictor for point/box prompts
        self.predictor = SamPredictor(self.sam)

    def _download_sam_checkpoint(self, model_type: str) -> str:
        """Download SAM checkpoint if not exists."""
        checkpoint_urls = {
            "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        }

        checkpoint_path = f"sam_{model_type}.pth"

        if not os.path.exists(checkpoint_path):
            print(f"Downloading SAM {model_type} checkpoint...")
            urlretrieve(checkpoint_urls[model_type], checkpoint_path)
            print(f"Downloaded to {checkpoint_path}")

        return checkpoint_path

    def generate_masks(self, image: np.ndarray) -> List[Dict]:
        """
        Generate masks for all objects in the image using automatic segmentation.

        Args:
            image: Input image as numpy array (RGB)

        Returns:
            List of mask dictionaries with segmentation info
        """
        try:
            masks = self.mask_generator.generate(image)
            print(f"Generated {len(masks)} masks for the image")
            return masks
        except Exception as e:
            print(f"Error generating masks: {e}")
            return []

    def predict_with_points(self, image: np.ndarray, points: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Generate masks using point prompts.

        Args:
            image: Input image as numpy array (RGB)
            points: Point coordinates [[x, y], ...]
            labels: Point labels (1 for foreground, 0 for background)

        Returns:
            Dictionary with masks, scores, and logits
        """
        self.predictor.set_image(image)
        masks, scores, logits = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )

        return {
            'masks': masks,
            'scores': scores,
            'logits': logits
        }


class SegmentProcessor:
    """Process segmented objects and generate embeddings."""

    def __init__(self, embedding_model: EmbeddingModel):
        """
        Initialize segment processor.

        Args:
            embedding_model: CLIP model for generating embeddings
        """
        self.embedding_model = embedding_model

    def isolate_segment(self, image: np.ndarray, mask: np.ndarray) -> Image.Image:
        """
        Isolate object by applying mask to create transparent RGBA image.

        Args:
            image: Original image as numpy array (RGB)
            mask: Binary mask as numpy array

        Returns:
            PIL Image with transparent background (RGBA)
        """
        # Ensure image is RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Create RGBA image
            rgba_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
            rgba_image[:, :, :3] = image  # Copy RGB channels
            rgba_image[:, :, 3] = mask.astype(np.uint8) * 255  # Alpha channel from mask
        else:
            raise ValueError("Input image must be RGB format")

        return Image.fromarray(rgba_image, 'RGBA')

    def get_bounding_box(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Get bounding box coordinates from mask."""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not np.any(rows) or not np.any(cols):
            return 0, 0, mask.shape[1], mask.shape[0]

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        return cmin, rmin, cmax + 1, rmax + 1

    def crop_to_object(self, rgba_image: Image.Image, mask: np.ndarray, padding: int = 10) -> Image.Image:
        """
        Crop RGBA image to object bounding box with padding.

        Args:
            rgba_image: RGBA image with transparent background
            mask: Binary mask for the object
            padding: Padding around the bounding box

        Returns:
            Cropped RGBA image
        """
        x1, y1, x2, y2 = self.get_bounding_box(mask)

        # Add padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(rgba_image.width, x2 + padding)
        y2 = min(rgba_image.height, y2 + padding)

        return rgba_image.crop((x1, y1, x2, y2))

    def create_composite_background(self, rgba_image: Image.Image, bg_color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
        """
        Create a composite image with solid background for CLIP processing.

        Args:
            rgba_image: RGBA image with transparency
            bg_color: Background color (RGB)

        Returns:
            RGB image with solid background
        """
        # Create background
        background = Image.new('RGB', rgba_image.size, bg_color)

        # Composite RGBA image onto background
        background.paste(rgba_image, mask=rgba_image.split()[3])  # Use alpha channel as mask

        return background

    def process_segment(self, image: np.ndarray, mask_info: Dict, crop_objects: bool = True) -> Tuple[Image.Image, np.ndarray]:
        """
        Process a single segment: isolate, crop, and generate embedding.

        Args:
            image: Original image as numpy array (RGB)
            mask_info: Mask information dictionary from SAM
            crop_objects: Whether to crop objects to bounding box

        Returns:
            Tuple of (processed_image, embedding_vector)
        """
        mask = mask_info['segmentation']

        # Isolate the segment
        rgba_image = self.isolate_segment(image, mask)

        # Crop to object if requested
        if crop_objects:
            rgba_image = self.crop_to_object(rgba_image, mask)

        # Create composite with white background for CLIP
        rgb_image = self.create_composite_background(rgba_image)

        # Save to temporary file for CLIP processing
        temp_path = "temp_segment.png"
        rgb_image.save(temp_path)

        try:
            # Generate embedding
            embedding = self.embedding_model.encode_image(temp_path)

            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

            return rgba_image, embedding

        except Exception as e:
            print(f"Error processing segment: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return rgba_image, None


class SearchLogic:
    """Search and matching logic for character identification."""

    def __init__(self, vector_database: VectorDatabase):
        """
        Initialize search logic.

        Args:
            vector_database: Database containing character embeddings
        """
        self.vector_db = vector_database
        self.embeddings_matrix = self.vector_db.get_embeddings_matrix()

    def calculate_similarity(self, query_embedding: np.ndarray, database_embeddings: np.ndarray = None) -> np.ndarray:
        """
        Calculate cosine similarity between query and database embeddings.

        Args:
            query_embedding: Query embedding vector
            database_embeddings: Database embeddings matrix (uses stored if None)

        Returns:
            Array of similarity scores
        """
        if database_embeddings is None:
            database_embeddings = self.embeddings_matrix

        if database_embeddings.size == 0:
            return np.array([])

        # Normalize embeddings
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        db_norms = database_embeddings / np.linalg.norm(database_embeddings, axis=1, keepdims=True)

        # Calculate cosine similarity
        similarities = np.dot(db_norms, query_norm)

        return similarities

    def find_best_match(self, query_embedding: np.ndarray, top_k: int = 1) -> List[Dict]:
        """
        Find the best matching character(s) for the query embedding.

        Args:
            query_embedding: Embedding of the segmented object
            top_k: Number of top matches to return

        Returns:
            List of match dictionaries with character info and scores
        """
        if query_embedding is None:
            return []

        similarities = self.calculate_similarity(query_embedding)

        if len(similarities) == 0:
            return []

        # Get top-k matches
        top_indices = np.argsort(similarities)[::-1][:top_k]

        matches = []
        for idx in top_indices:
            entry = self.vector_db.data[idx]
            matches.append({
                'character': entry['label'],
                'similarity_score': float(similarities[idx]),
                'database_entry_id': idx,  # Use index instead of non-existent 'id' field
                'image_path': entry['image_path'],
                'metadata': entry.get('metadata', {})  # Use .get() to avoid KeyError
            })

        return matches

    def search_segments(self, segment_embeddings: List[np.ndarray], confidence_threshold: float = 0.7) -> List[Dict]:
        """
        Search for character matches across multiple segments.

        Args:
            segment_embeddings: List of embedding vectors from segments
            confidence_threshold: Minimum similarity score for valid matches

        Returns:
            List of character predictions for each segment
        """
        results = []

        for i, embedding in enumerate(segment_embeddings):
            if embedding is None:
                results.append({
                    'segment_id': i,
                    'prediction': 'Unknown',
                    'confidence': 0.0,
                    'matches': []
                })
                continue

            matches = self.find_best_match(embedding, top_k=3)

            if matches and matches[0]['similarity_score'] >= confidence_threshold:
                prediction = matches[0]['character']
                confidence = matches[0]['similarity_score']
            else:
                prediction = 'Unknown'
                confidence = matches[0]['similarity_score'] if matches else 0.0

            results.append({
                'segment_id': i,
                'prediction': prediction,
                'confidence': confidence,
                'matches': matches
            })

        return results


class SceneAnalysisPipeline:
    """Main scene analysis pipeline orchestrating segmentation, processing, and search."""

    def __init__(self, vector_database: VectorDatabase, sam_model_type: str = "vit_h"):
        """
        Initialize the scene analysis pipeline.

        Args:
            vector_database: Pre-built database with character embeddings
            sam_model_type: SAM model variant to use
        """
        print("Initializing Scene Analysis Pipeline...")

        # Initialize components
        self.embedding_model = EmbeddingModel()
        self.segmentation_model = SegmentationModel(sam_model_type)
        self.segment_processor = SegmentProcessor(self.embedding_model)
        self.search_logic = SearchLogic(vector_database)

        print("Scene Analysis Pipeline ready!")

    def analyze_scene(self, image_path: str, confidence_threshold: float = 0.7,
                     save_visualizations: bool = True) -> Dict:
        """
        Analyze a scene image to identify characters.

        Args:
            image_path: Path to the input image
            confidence_threshold: Minimum confidence for character predictions
            save_visualizations: Whether to save visualization images

        Returns:
            Dictionary with analysis results
        """
        print(f"Analyzing scene: {image_path}")

        # Load image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Generate masks
        print("Generating object masks...")
        masks = self.segmentation_model.generate_masks(image_rgb)

        if not masks:
            return {
                'image_path': image_path,
                'total_segments': 0,
                'characters_found': [],
                'processing_error': 'No masks generated'
            }

        # Process segments
        print(f"Processing {len(masks)} segments...")
        segment_results = []
        segment_embeddings = []
        processed_images = []

        for i, mask_info in enumerate(masks):
            try:
                processed_img, embedding = self.segment_processor.process_segment(image_rgb, mask_info)
                processed_images.append(processed_img)
                segment_embeddings.append(embedding)

                segment_results.append({
                    'segment_id': i,
                    'area': mask_info['area'],
                    'bbox': mask_info['bbox'],
                    'stability_score': mask_info['stability_score']
                })

            except Exception as e:
                print(f"Error processing segment {i}: {e}")
                processed_images.append(None)
                segment_embeddings.append(None)

        # Search for character matches
        print("Searching for character matches...")
        search_results = self.search_logic.search_segments(segment_embeddings, confidence_threshold)

        # Combine results
        characters_found = []
        for segment_result, search_result in zip(segment_results, search_results):
            if search_result['prediction'] != 'Unknown':
                characters_found.append({
                    'character': search_result['prediction'],
                    'confidence': search_result['confidence'],
                    'segment_info': segment_result,
                    'matches': search_result['matches']
                })

        # Save visualizations
        if save_visualizations:
            self._save_visualizations(image_rgb, masks, search_results,
                                    os.path.splitext(os.path.basename(image_path))[0])

        result = {
            'image_path': image_path,
            'total_segments': len(masks),
            'characters_found': characters_found,
            'all_predictions': search_results,
            'unique_characters': list(set([c['character'] for c in characters_found]))
        }

        print(f"Analysis complete! Found {len(characters_found)} character instances from {len(result['unique_characters'])} unique characters")

        return result

    def _save_visualizations(self, image: np.ndarray, masks: List[Dict],
                           predictions: List[Dict], base_name: str):
        """Save visualization images showing segmentation and predictions."""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(15, 7))

            # Original image with masks
            axes[0].imshow(image)
            axes[0].set_title('Segmented Objects')
            axes[0].axis('off')

            # Draw masks with predictions
            for i, (mask_info, pred) in enumerate(zip(masks, predictions)):
                mask = mask_info['segmentation']
                color = np.random.random(3)

                # Show mask
                axes[0].contour(mask, colors=[color], linewidths=2)

                # Add prediction label
                bbox = mask_info['bbox']
                x, y = bbox[0], bbox[1]

                label = f"{pred['prediction']}\n({pred['confidence']:.2f})"
                axes[0].text(x, y, label, fontsize=8, color='white',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))

            # Summary statistics
            axes[1].axis('off')
            summary_text = f"Scene Analysis Results\n\n"
            summary_text += f"Total segments: {len(masks)}\n"

            char_counts = {}
            for pred in predictions:
                char = pred['prediction']
                char_counts[char] = char_counts.get(char, 0) + 1

            summary_text += "\nCharacter Detections:\n"
            for char, count in char_counts.items():
                if char != 'Unknown':
                    summary_text += f"• {char}: {count}\n"

            axes[1].text(0.1, 0.9, summary_text, fontsize=12, verticalalignment='top',
                        transform=axes[1].transAxes)

            # Save visualization
            output_path = f"analysis_{base_name}.png"
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"Visualization saved as: {output_path}")

        except Exception as e:
            print(f"Error saving visualization: {e}")


if __name__ == "__main__":
    # Example usage
    try:
        # Load pre-built vector database
        from indexing_pipeline import VectorDatabase

        vector_db = VectorDatabase()
        vector_db.load_from_file("naruto_character_embeddings.pkl")

        # Initialize scene analysis pipeline
        pipeline = SceneAnalysisPipeline(vector_db, sam_model_type="vit_b")

        # Analyze a sample image (you can replace with your own image)
        sample_image = r"C:\Users\Utente\PycharmProjects\DL_Exam_v2\Anime -Naruto-.v1i.multiclass\test\43596_jpg.rf.5b5721f1535fb8fb376f60ee2335807a.jpg"

        if os.path.exists(sample_image):
            results = pipeline.analyze_scene(sample_image)

            print("\n=== Scene Analysis Results ===")
            print(f"Image: {results['image_path']}")
            print(f"Total segments found: {results['total_segments']}")
            print(f"Characters detected: {len(results['characters_found'])}")
            print(f"Unique characters: {results['unique_characters']}")

            for char_detection in results['characters_found']:
                print(f"\n• Character: {char_detection['character']}")
                print(f"  Confidence: {char_detection['confidence']:.3f}")
                print(f"  Segment area: {char_detection['segment_info']['area']}")
        else:
            print(f"Sample image not found: {sample_image}")

    except Exception as e:
        print(f"Error in demo: {e}")
