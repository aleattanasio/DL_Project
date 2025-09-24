"""
Gradio User Interface for Naruto Character Recognition Pipeline
Provides Database Query and Scene Analysis functionalities
"""

import gradio as gr
import torch
import numpy as np
import os
import tempfile
import shutil
from PIL import Image
import matplotlib.pyplot as plt
import io
import base64
from typing import List, Tuple, Dict, Optional

# Import our pipeline components
from indexing_pipeline import VectorDatabase, EmbeddingModel
from scene_analysis_pipeline import SceneAnalysisPipeline


class NarutoCharacterUI:
    """Main UI class for Naruto Character Recognition System."""

    def __init__(self):
        """Initialize the UI with multiple model support."""
        self.available_models = {
            "Pre-trained CLIP": {
                "model_path": None,
                "database_path": "naruto_embeddings.pkl",
                "description": "Original CLIP model without fine-tuning",
                "model_type": "clip"
            },
            "Fine-tuned CLIP (Pre-trained DB)": {
                "model_path": "results_clip_finetuned/checkpoints/best_model.pth",
                "database_path": "naruto_embeddings.pkl",
                "description": "Fine-tuned CLIP model using pre-trained embeddings database",
                "model_type": "clip"
            },
            "Fine-tuned CLIP (Fine-tuned DB)": {
                "model_path": "results_clip_finetuned/checkpoints/best_model.pth",
                "database_path": "results_clip_finetuned/databases/naruto_finetuned_embeddings.pkl",
                "description": "Fine-tuned CLIP model using fine-tuned embeddings database",
                "model_type": "clip"
            },
            "FAISS Fine-tuned CLIP": {
                "model_path": "results_faiss_clip_finetuned/checkpoints/best_model.pth",
                "database_path": "results_faiss_clip_finetuned/databases/naruto_finetuned_faiss_database.pkl",
                "description": "Fine-tuned CLIP model with FAISS vector database",
                "model_type": "clip"
            },
            "Fine-tuned BLIP-2": {
                "model_path": "results_blip2_finetuned/checkpoints/best_model.pth",
                "database_path": "results_blip2_finetuned/databases/naruto_blip2_finetuned_embeddings.pkl",
                "description": "Fine-tuned BLIP-2 model using fine-tuned embeddings database",
                "model_type": "blip2",
                "blip2_model_name": "Salesforce/blip2-opt-2.7b"
            }
        }

        self.current_model_name = "Pre-trained CLIP"  # Default model
        self.vector_db = None
        self.faiss_db = None
        self.embedding_model = None
        self.scene_pipeline = None

        # Initialize with default model
        self._switch_model(self.current_model_name)

    def _switch_model(self, model_name: str):
        """Switch to a different enhanced CLIP model."""
        if model_name not in self.available_models:
            return f"❌ Model '{model_name}' not available"

        try:
            print(f"🔄 Switching to model: {model_name}")
            model_config = self.available_models[model_name]

            self.current_model_name = model_name
            self.database_path = model_config["database_path"]
            self.fine_tuned_model_path = model_config["model_path"]
            self.use_fine_tuned_model = model_config["model_path"] is not None

            # Reload database and models
            self._load_database()
            self._initialize_models()

            return f"✅ Successfully switched to: {model_name}"

        except Exception as e:
            return f"❌ Error switching model: {str(e)}"

    def _load_database(self):
        """Load the character database with support for both traditional and FAISS databases."""
        try:
            # Try to load FAISS database first (more advanced)
            if "FAISS" in self.current_model_name and self.use_fine_tuned_model and os.path.exists(self.database_path):
                try:
                    from faiss_clip_finetuning import FAISSVectorDatabase
                    self.faiss_db = FAISSVectorDatabase(dimension=512)
                    self.faiss_db.load_from_file(self.database_path)
                    print(f"✓ Loaded FAISS database with {len(self.faiss_db.data)} fine-tuned character embeddings")

                    # Create compatibility layer for existing interface
                    self._create_faiss_compatibility_layer()
                    return
                except Exception as e:
                    print(f"Warning: Could not load FAISS database: {e}")
                    print("Falling back to traditional database...")

            # Check if it's a fine-tuned database (pickle format with specific structure)
            if "Fine-tuned DB" in self.current_model_name and os.path.exists(self.database_path):
                try:
                    import pickle
                    with open(self.database_path, 'rb') as f:
                        database_data = pickle.load(f)

                    # Check if it's the fine-tuned database format
                    if 'data' in database_data and 'embeddings_matrix' in database_data:
                        print(f"✓ Loaded fine-tuned embeddings database with {len(database_data['data'])} character images")
                        self._create_finetuned_compatibility_layer(database_data)
                        return
                except Exception as e:
                    print(f"Warning: Could not load fine-tuned database: {e}")
                    print("Falling back to traditional database...")

            # Fallback to traditional database
            if os.path.exists(self.database_path):
                self.vector_db = VectorDatabase()
                self.vector_db.load_from_file(self.database_path)
                print(f"✓ Loaded traditional database with {len(self.vector_db.data)} character images")
            else:
                raise FileNotFoundError(f"No database found: {self.database_path}")

        except Exception as e:
            print(f"Error loading database: {e}")
            self.vector_db = None
            self.faiss_db = None

    def _create_faiss_compatibility_layer(self):
        """Create compatibility layer for FAISS database to work with existing interface."""
        if self.faiss_db:
            # Create a mock vector_db object for compatibility
            class FAISSCompatibility:
                def __init__(self, faiss_db):
                    self.faiss_db = faiss_db
                    self.data = self._convert_metadata_to_data()

                def _convert_metadata_to_data(self):
                    """Convert FAISS metadata to traditional format."""
                    data = []
                    # FAISSVectorDatabase uses self.data instead of self.metadata
                    for i, item in enumerate(self.faiss_db.data):
                        data.append({
                            'image_path': item['image_path'],
                            'label': item['label'],
                            'embedding': None  # Will be retrieved from FAISS when needed
                        })
                    return data

                def get_embeddings_matrix(self):
                    """Get embeddings matrix from FAISS index."""
                    return self.faiss_db.index.reconstruct_n(0, self.faiss_db.index.ntotal)

                def get_stats(self):
                    """Get database statistics."""
                    labels = [item['label'] for item in self.data]
                    label_counts = {}
                    for label in labels:
                        label_counts[label] = label_counts.get(label, 0) + 1

                    return {
                        'total_entries': len(self.data),
                        'unique_labels': len(set(labels)),
                        'embedding_dimension': self.faiss_db.dimension,
                        'label_distribution': label_counts
                    }

            self.vector_db = FAISSCompatibility(self.faiss_db)

    def _create_finetuned_compatibility_layer(self, database_data):
        """Create compatibility layer for fine-tuned database to work with existing interface."""
        class FineTunedCompatibility:
            def __init__(self, database_data):
                self.data = database_data['data']
                self.embeddings_matrix = database_data['embeddings_matrix']
                self.model_info = database_data.get('model_info', {})

            def get_embeddings_matrix(self):
                """Get embeddings matrix."""
                return self.embeddings_matrix

            def get_stats(self):
                """Get database statistics."""
                labels = [item['label'] for item in self.data]
                label_counts = {}
                for label in labels:
                    label_counts[label] = label_counts.get(label, 0) + 1

                return {
                    'total_entries': len(self.data),
                    'unique_labels': len(set(labels)),
                    'embedding_dimension': self.embeddings_matrix.shape[1] if len(self.embeddings_matrix) > 0 else 512,
                    'label_distribution': label_counts,
                    'fine_tuned': self.model_info.get('fine_tuned', False)
                }

        self.vector_db = FineTunedCompatibility(database_data)

    def _initialize_models(self):
        """Initialize CLIP, BLIP-2 and SAM models with support for fine-tuned models."""
        try:
            if self.vector_db:
                model_config = self.available_models[self.current_model_name]
                model_type = model_config.get("model_type", "clip")

                # Initialize embedding model based on type
                if model_type == "blip2":
                    self.embedding_model = self._load_blip2_model(model_config)
                    print(f"✓ Loaded BLIP-2 model: {model_config.get('blip2_model_name', 'default')}")
                elif self.use_fine_tuned_model and os.path.exists(self.fine_tuned_model_path):
                    self.embedding_model = self._load_fine_tuned_model()
                    print("✓ Loaded fine-tuned CLIP model")
                else:
                    self.embedding_model = EmbeddingModel()
                    print("✓ Loaded pre-trained CLIP model")

                # Initialize scene analysis pipeline with better SAM model detection
                sam_model_paths = ["sam_vit_h.pth", "sam_vit_h_4b8939.pth", "sam_vit_h.pth"]
                sam_model_found = any(os.path.exists(path) for path in sam_model_paths)

                if sam_model_found:
                    try:
                        self.scene_pipeline = SceneAnalysisPipeline(self.vector_db, sam_model_type="vit_h")
                        print("✓ Scene analysis pipeline initialized with SAM model")
                    except Exception as e:
                        print(f"⚠️ Error initializing scene analysis: {e}")
                        print("   Scene analysis will not be available")
                        self.scene_pipeline = None
                else:
                    print(f"⚠️ SAM model not found")
                    print("   Checked paths: " + ", ".join(sam_model_paths))
                    print("   Scene analysis will not be available")
                    print("   To enable scene analysis, download the SAM model:")
                    print("   https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
                    print("   and save it as 'sam_vit_h.pth' in the project root")
                    self.scene_pipeline = None
        except Exception as e:
            print(f"Error initializing models: {e}")
            self.scene_pipeline = None

    def _load_fine_tuned_model(self):
        """Load the fine-tuned CLIP model."""
        try:
            import clip

            # Load checkpoint with PyTorch 2.6 compatibility
            checkpoint = torch.load(self.fine_tuned_model_path, map_location='cpu', weights_only=False)

            # Get model name from checkpoint
            model_name = checkpoint['hyperparameters']['model_name']

            # Load CLIP model
            model, preprocess = clip.load(model_name, device='cuda' if torch.cuda.is_available() else 'cpu')

            # Load fine-tuned weights
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            # Create enhanced embedding model
            class FineTunedEmbeddingModel:
                def __init__(self, model, preprocess, device):
                    self.model = model
                    self.preprocess = preprocess
                    self.device = device

                def encode_image(self, image_path):
                    """Encode image using fine-tuned model."""
                    try:
                        from PIL import Image
                        image = Image.open(image_path).convert('RGB')
                        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

                        with torch.no_grad():
                            features = self.model.encode_image(image_tensor)
                            features = features / features.norm(dim=-1, keepdim=True)

                        return features.cpu().numpy().flatten()
                    except Exception as e:
                        print(f"Error encoding image: {e}")
                        return None

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            return FineTunedEmbeddingModel(model, preprocess, device)

        except Exception as e:
            print(f"Error loading fine-tuned model: {e}")
            # Fallback to regular model
            return EmbeddingModel()

    def _load_blip2_model(self, model_config):
        """Load the BLIP-2 model (pre-trained or fine-tuned)."""
        try:
            from transformers import Blip2Processor, Blip2Model

            model_name = model_config["blip2_model_name"]
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            # Load processor and model
            processor = Blip2Processor.from_pretrained(model_name, use_fast=True)

            if self.use_fine_tuned_model and os.path.exists(self.fine_tuned_model_path):
                # Load fine-tuned BLIP-2 model
                print("Loading fine-tuned BLIP-2 model...")

                # Load the base model first
                base_model = Blip2Model.from_pretrained(model_name, torch_dtype=torch.float16)

                # Load fine-tuned checkpoint
                checkpoint = torch.load(self.fine_tuned_model_path, map_location='cpu', weights_only=False)

                # Create the same model architecture as in training
                from blip2_finetuning import BLIP2FineTuner
                fine_tuner = BLIP2FineTuner(model_name=model_name)
                fine_tuner.num_classes = len(checkpoint['class_names'])
                fine_tuner.class_names = checkpoint['class_names']
                model = fine_tuner.create_model()

                # Load the fine-tuned weights
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(device)
                model.eval()

                # Create wrapper for fine-tuned model
                class FineTunedBLIP2EmbeddingModel:
                    def __init__(self, model, processor, device):
                        self.model = model
                        self.processor = processor
                        self.device = device

                    def encode_image(self, image_path):
                        """Encode image using fine-tuned BLIP-2 model."""
                        try:
                            image = Image.open(image_path).convert('RGB')
                            inputs = self.processor(images=image, return_tensors="pt")
                            pixel_values = inputs['pixel_values'].to(self.device)

                            with torch.no_grad():
                                # Extract features from the fine-tuned model's base_model
                                outputs = self.model.base_model.get_qformer_features(pixel_values=pixel_values)
                                features = outputs.last_hidden_state.mean(dim=1)  # Pool features

                                # Normalize embeddings
                                features = features / features.norm(dim=-1, keepdim=True)

                            return features.cpu().numpy().flatten()
                        except Exception as e:
                            print(f"Error encoding image with fine-tuned BLIP-2: {e}")
                            return None

                    def encode_text(self, text):
                        """Encode text using fine-tuned BLIP-2 model."""
                        try:
                            inputs = self.processor(text=text, return_tensors="pt")
                            inputs = {k: v.to(self.device) for k, v in inputs.items()}

                            with torch.no_grad():
                                outputs = self.model.base_model.get_qformer_features(**inputs)
                                features = outputs.last_hidden_state.mean(dim=1)
                                features = features / features.norm(dim=-1, keepdim=True)

                            return features.cpu().numpy().flatten()
                        except Exception as e:
                            print(f"Error encoding text with fine-tuned BLIP-2: {e}")
                            return None

                return FineTunedBLIP2EmbeddingModel(model, processor, device)

            else:
                # Load pre-trained BLIP-2 model
                print("Loading pre-trained BLIP-2 model...")
                model = Blip2Model.from_pretrained(model_name, torch_dtype=torch.float16)
                model.to(device)
                model.eval()

                # Create wrapper for pre-trained model
                class PreTrainedBLIP2EmbeddingModel:
                    def __init__(self, model, processor, device):
                        self.model = model
                        self.processor = processor
                        self.device = device

                    def encode_image(self, image_path):
                        """Encode image using pre-trained BLIP-2 model."""
                        try:
                            image = Image.open(image_path).convert('RGB')
                            inputs = self.processor(images=image, return_tensors="pt")
                            inputs = {k: v.to(self.device) for k, v in inputs.items()}

                            with torch.no_grad():
                                outputs = self.model.get_qformer_features(**inputs)
                                features = outputs.last_hidden_state.mean(dim=1)
                                features = features / features.norm(dim=-1, keepdim=True)

                            return features.cpu().numpy().flatten()
                        except Exception as e:
                            print(f"Error encoding image with pre-trained BLIP-2: {e}")
                            return None

                    def encode_text(self, text):
                        """Encode text using pre-trained BLIP-2 model."""
                        try:
                            inputs = self.processor(text=text, return_tensors="pt")
                            inputs = {k: v.to(self.device) for k, v in inputs.items()}

                            with torch.no_grad():
                                outputs = self.model.get_qformer_features(**inputs)
                                features = outputs.last_hidden_state.mean(dim=1)
                                features = features / features.norm(dim=-1, keepdim=True)

                            return features.cpu().numpy().flatten()
                        except Exception as e:
                            print(f"Error encoding text with pre-trained BLIP-2: {e}")
                            return None

                return PreTrainedBLIP2EmbeddingModel(model, processor, device)

        except Exception as e:
            print(f"Error loading BLIP-2 model: {e}")
            # Fallback to CLIP if BLIP-2 fails
            return EmbeddingModel()

    def query_by_image(self, query_image: Image.Image, top_k: int = 5) -> Tuple[str, List[Image.Image]]:
        """
        Query database using an uploaded image.

        Args:
            query_image: PIL Image uploaded by user
            top_k: Number of similar images to return

        Returns:
            Tuple of (results_text, list_of_similar_images)
        """
        if not query_image or not self.vector_db or not self.embedding_model:
            return "❌ Error: Database or models not loaded properly", []

        try:
            # Save temporary image for processing
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                query_image.save(tmp.name)
                temp_path = tmp.name

            # Generate embedding for query image
            query_embedding = self.embedding_model.encode_image(temp_path)

            # Clean up temp file
            os.unlink(temp_path)

            if query_embedding is None:
                return "❌ Error: Could not process the uploaded image", []

            # Calculate similarities
            embeddings_matrix = self.vector_db.get_embeddings_matrix()
            similarities = np.dot(embeddings_matrix, query_embedding.T).flatten()

            # Get top-k matches
            top_indices = np.argsort(similarities)[::-1][:top_k]

            # Prepare results
            results_text = f"🔍 **Query Results** (Top {top_k} matches)\n\n"
            similar_images = []

            for i, idx in enumerate(top_indices):
                entry = self.vector_db.data[idx]
                similarity = similarities[idx]
                character = entry['label']
                image_path = entry['image_path']

                results_text += f"**{i+1}.** {character} - Similarity: {similarity:.3f}\n"
                results_text += f"   File: {os.path.basename(image_path)}\n\n"

                # Load and add image
                if os.path.exists(image_path):
                    similar_images.append(Image.open(image_path))
                else:
                    # Create placeholder if image not found
                    placeholder = Image.new('RGB', (224, 224), color='gray')
                    similar_images.append(placeholder)

            return results_text, similar_images

        except Exception as e:
            return f"❌ Error processing image: {str(e)}", []

    def query_by_text(self, text_query: str, top_k: int = 5) -> Tuple[str, List[Image.Image]]:
        """
        Query database using text description.

        Args:
            text_query: Text description of character
            top_k: Number of similar images to return

        Returns:
            Tuple of (results_text, list_of_similar_images)
        """
        if not text_query.strip() or not self.vector_db or not self.embedding_model:
            return "❌ Error: Please enter a text description", []

        try:
            # Check if current model is BLIP-2
            model_config = self.available_models[self.current_model_name]
            model_type = model_config.get("model_type", "clip")

            if model_type == "blip2":
                # Use BLIP-2 for text encoding
                text_embedding = self.embedding_model.encode_text(text_query)
                if text_embedding is None:
                    return "❌ Error: Could not process the text query with BLIP-2", []
            else:
                # Use CLIP for text encoding
                import clip
                device = self.embedding_model.device
                model = self.embedding_model.model

                # Tokenize and encode text
                text_tokens = clip.tokenize([text_query]).to(device)
                with torch.no_grad():
                    text_features = model.encode_text(text_tokens)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    text_embedding = text_features.cpu().numpy().flatten()

            # Calculate similarities with image embeddings
            embeddings_matrix = self.vector_db.get_embeddings_matrix()
            similarities = np.dot(embeddings_matrix, text_embedding.T).flatten()

            # Get top-k matches
            top_indices = np.argsort(similarities)[::-1][:top_k]

            # Prepare results
            model_name = model_type.upper()
            results_text = f"🔍 **{model_name} Text Query Results** for: \"{text_query}\"\n\n"
            similar_images = []

            for i, idx in enumerate(top_indices):
                entry = self.vector_db.data[idx]
                similarity = similarities[idx]
                character = entry['label']
                image_path = entry['image_path']

                results_text += f"**{i+1}.** {character} - Similarity: {similarity:.3f}\n"
                results_text += f"   File: {os.path.basename(image_path)}\n\n"

                # Load and add image
                if os.path.exists(image_path):
                    similar_images.append(Image.open(image_path))
                else:
                    placeholder = Image.new('RGB', (224, 224), color='gray')
                    similar_images.append(placeholder)

            return results_text, similar_images

        except Exception as e:
            return f"❌ Error processing text query: {str(e)}", []

    def analyze_scene(self, scene_image: Image.Image, confidence_threshold: float = 0.6) -> Tuple[str, Image.Image]:
        """
        Analyze a complex scene to identify characters.

        Args:
            scene_image: PIL Image of the scene
            confidence_threshold: Minimum confidence for character detection

        Returns:
            Tuple of (analysis_text, visualization_image)
        """
        if not scene_image:
            return "❌ Error: Please upload an image for scene analysis", None

        if not self.scene_pipeline:
            error_msg = "⚠️ **Scene Analysis Not Available**\n\n"
            error_msg += "The SAM (Segment Anything Model) is required for scene analysis but is not available.\n\n"
            error_msg += "**To enable scene analysis:**\n"
            error_msg += "1. Download the SAM model from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth\n"
            error_msg += "2. Save it as `sam_vit_h.pth` in the project root directory\n"
            error_msg += "3. Restart the application\n\n"
            error_msg += "**Alternative:** You can still use the Database Query feature to search for individual characters!"

            # Create a placeholder image with instructions
            try:
                import matplotlib.pyplot as plt
                import numpy as np

                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                ax.text(0.5, 0.5, 'SAM Model Required\n\nScene Analysis Unavailable\n\nPlease download sam_vit_h.pth\nto enable this feature',
                       horizontalalignment='center', verticalalignment='center',
                       fontsize=16, fontweight='bold', color='red',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                ax.set_title('Scene Analysis - SAM Model Missing', fontsize=18, fontweight='bold')

                # Convert to PIL Image
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                placeholder_image = Image.open(buf)
                plt.close()

                return error_msg, placeholder_image

            except Exception:
                # If matplotlib fails, return the original image
                return error_msg, scene_image

        try:
            # Save temporary image for processing
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                scene_image.save(tmp.name)
                temp_path = tmp.name

            # Analyze the scene
            results = self.scene_pipeline.analyze_scene(
                temp_path,
                confidence_threshold=confidence_threshold,
                save_visualizations=False  # We'll create our own visualization
            )

            # Clean up temp file
            os.unlink(temp_path)

            # Create analysis text
            analysis_text = f"🎭 **Scene Analysis Results**\n\n"
            analysis_text += f"📊 **Statistics:**\n"
            analysis_text += f"• Total segments detected: {results['total_segments']}\n"
            analysis_text += f"• Characters identified: {len(results['characters_found'])}\n"
            analysis_text += f"• Unique characters: {len(results['unique_characters'])}\n\n"

            if results['characters_found']:
                analysis_text += f"🎯 **Character Detections:**\n"

                # Group by character
                char_detections = {}
                for detection in results['characters_found']:
                    char = detection['character']
                    if char not in char_detections:
                        char_detections[char] = []
                    char_detections[char].append(detection)

                for char, detections in char_detections.items():
                    analysis_text += f"\n**{char}** ({len(detections)} instances):\n"
                    for i, det in enumerate(detections[:3]):  # Show top 3 instances
                        conf = det['confidence']
                        area = det['segment_info']['area']
                        analysis_text += f"  {i+1}. Confidence: {conf:.3f}, Area: {area} pixels\n"
                    if len(detections) > 3:
                        analysis_text += f"  ... and {len(detections) - 3} more\n"
            else:
                analysis_text += "⚠️ No characters detected above the confidence threshold.\n"
                analysis_text += "Try lowering the confidence threshold or using a different image."

            # Create visualization
            visualization = self._create_scene_visualization(scene_image, results)

            return analysis_text, visualization

        except Exception as e:
            return f"❌ Error analyzing scene: {str(e)}", None

    def _create_scene_visualization(self, original_image: Image.Image, results: Dict) -> Image.Image:
        """Create a visualization of the scene analysis results."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            from matplotlib import colors
            import numpy as np

            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

            # Convert PIL to numpy
            img_array = np.array(original_image)

            # Original image
            ax1.imshow(img_array)
            ax1.set_title('Original Scene', fontsize=14, fontweight='bold')
            ax1.axis('off')

            # Scene with detections
            ax2.imshow(img_array)
            ax2.set_title('Character Detections', fontsize=14, fontweight='bold')
            ax2.axis('off')

            # Add bounding boxes and labels
            character_colors = {
                'Naruto': 'orange',
                'Sakura': 'pink',
                'Tsunade': 'yellow',
                'Gara': 'red',
                'Unknown': 'gray'
            }

            for detection in results.get('characters_found', []):
                char = detection['character']
                conf = detection['confidence']
                bbox = detection['segment_info']['bbox']

                # bbox format: [x, y, width, height]
                x, y, w, h = bbox

                # Get color for character
                color = character_colors.get(char, 'blue')

                # Add rectangle
                rect = patches.Rectangle((x, y), w, h, linewidth=2,
                                       edgecolor=color, facecolor='none')
                ax2.add_patch(rect)

                # Add label
                label = f"{char}\n{conf:.2f}"
                ax2.text(x, y-5, label, fontsize=10, color=color,
                        fontweight='bold', bbox=dict(boxstyle="round,pad=0.3",
                        facecolor='white', alpha=0.8))

            # Add legend
            legend_elements = []
            for char, color in character_colors.items():
                if any(d['character'] == char for d in results.get('characters_found', [])):
                    legend_elements.append(patches.Patch(color=color, label=char))

            if legend_elements:
                ax2.legend(handles=legend_elements, loc='upper right')

            plt.tight_layout()

            # Convert matplotlib figure to PIL Image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            visualization = Image.open(buf)
            plt.close()

            return visualization

        except Exception as e:
            print(f"Error creating visualization: {e}")
            # Return original image if visualization fails
            return original_image

    def get_database_stats(self) -> str:
        """Get database statistics for display."""
        if not self.vector_db:
            return "❌ Database not loaded"

        stats = self.vector_db.get_stats()

        stats_text = f"📊 **Database Statistics**\n\n"
        stats_text += f"• Total character images: **{stats['total_entries']}**\n"
        stats_text += f"• Unique character classes: **{stats['unique_labels']}**\n"
        stats_text += f"• Embedding dimensions: **{stats['embedding_dimension']}D**\n\n"

        stats_text += f"**Character Distribution:**\n"
        for char, count in sorted(stats['label_distribution'].items(),
                                key=lambda x: x[1], reverse=True):
            percentage = (count / stats['total_entries']) * 100
            stats_text += f"• {char}: {count} images ({percentage:.1f}%)\n"

        return stats_text

    def get_model_info(self) -> str:
        """Get information about available models."""
        info = f"🤖 **Current Model**: {self.current_model_name}\n\n"
        info += "📋 **Available Models**:\n\n"

        for name, config in self.available_models.items():
            # Fix the syntax error with proper conditional logic
            if config["model_path"] is None:
                status = "🟢"  # Pre-trained model is always available
            else:
                status = "🟢" if os.path.exists(config["model_path"]) else "🔴"

            current = " *(Current)*" if name == self.current_model_name else ""
            info += f"- **{name}**{current} {status}\n"
            info += f"  - {config['description']}\n"
            if config["model_path"]:
                info += f"  - Model: `{config['model_path']}`\n"
            info += f"  - Database: `{config['database_path']}`\n\n"

        # Add database statistics
        if self.vector_db:
            stats = self.vector_db.get_stats() if hasattr(self.vector_db, 'get_stats') else {}
            if stats:
                info += f"📊 **Current Database Stats**:\n"
                info += f"- Total Images: {stats.get('total_entries', 'N/A')}\n"
                info += f"- Characters: {stats.get('unique_labels', 'N/A')}\n"
                info += f"- Embedding Dimension: {stats.get('embedding_dimension', 'N/A')}\n"

        return info


def create_gradio_interface():
    """Create and configure the Gradio interface."""

    # Initialize the UI class
    ui = NarutoCharacterUI()

    # Custom CSS for better appearance
    css = """
    .gradio-container {
        max-width: 1200px !important;
    }
    .tab-nav {
        font-size: 16px !important;
    }
    .gr-button {
        background: linear-gradient(45deg, #ff6b35, #f7931e) !important;
        border: none !important;
        color: white !important;
    }
    .gr-button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(255, 107, 53, 0.4) !important;
    }
    """

    # Create the main interface
    with gr.Blocks(css=css, title="Naruto Character Recognition System") as interface:

        # Header
        gr.Markdown("""
        # 🍥 Naruto Character Recognition System
        
        Welcome to the AI-powered Naruto character recognition system! This application uses advanced computer vision 
        techniques including CLIP embeddings and SAM segmentation to identify characters from the Naruto anime series.
        
        Choose a tab below to get started:
        """)

        with gr.Tabs():

            # Database Query Tab
            with gr.Tab("🔍 Database Query", id="query_tab"):
                gr.Markdown("""
                ### Find Similar Characters
                Upload an image or describe a character to find the most similar images from our database.
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        # Model Selection for Database Query
                        gr.Markdown("### 🤖 Model Selection")
                        query_model_dropdown = gr.Dropdown(
                            choices=list(ui.available_models.keys()),
                            value=ui.current_model_name,
                            label="AI Model Selection",
                            info="Choose which AI model to use for database queries (CLIP or BLIP-2)"
                        )

                        query_model_switch_btn = gr.Button("🔄 Switch Model", variant="secondary")
                        query_model_status = gr.Markdown(f"✅ Current: {ui.current_model_name}")

                        # Database stats
                        stats_display = gr.Markdown(
                            value=ui.get_database_stats(),
                            label="Database Information"
                        )

                        # Model info
                        model_info_display = gr.Markdown(
                            value=ui.get_model_info(),
                            label="Model Information"
                        )

                        # Query controls
                        gr.Markdown("### Query Options")

                        query_image = gr.Image(
                            type="pil",
                            label="Upload Character Image",
                            height=300
                        )

                        gr.Markdown("**OR**")

                        query_text = gr.Textbox(
                            label="Describe Character",
                            placeholder="e.g., 'ninja with orange outfit' or 'pink hair kunoichi'",
                            lines=2
                        )

                        top_k_slider = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=5,
                            step=1,
                            label="Number of Results"
                        )

                        with gr.Row():
                            search_image_btn = gr.Button("🖼️ Search by Image", variant="primary")
                            search_text_btn = gr.Button("📝 Search by Text", variant="primary")

                    with gr.Column(scale=2):
                        query_results = gr.Markdown(
                            value="Upload an image or enter a text description to search the database.",
                            label="Search Results"
                        )

                        similar_images = gr.Gallery(
                            label="Similar Characters",
                            columns=3,
                            rows=2,
                            height="auto"
                        )

                # Event handlers for database query
                def switch_query_model(model_name):
                    status = ui._switch_model(model_name)
                    stats = ui.get_database_stats()
                    model_info = ui.get_model_info()
                    return status, stats, model_info

                query_model_switch_btn.click(
                    switch_query_model,
                    inputs=[query_model_dropdown],
                    outputs=[query_model_status, stats_display, model_info_display]
                )

                search_image_btn.click(
                    ui.query_by_image,
                    inputs=[query_image, top_k_slider],
                    outputs=[query_results, similar_images]
                )

                search_text_btn.click(
                    ui.query_by_text,
                    inputs=[query_text, top_k_slider],
                    outputs=[query_results, similar_images]
                )

            # Scene Analysis Tab
            with gr.Tab("🎭 Scene Analysis", id="analysis_tab"):
                gr.Markdown("""
                ### Analyze Complex Scenes
                Upload an image containing multiple characters or complex scenes. The system will automatically 
                segment objects and identify any Naruto characters present.
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        # Model Selection for Scene Analysis
                        gr.Markdown("### 🤖 Model Selection")
                        scene_model_dropdown = gr.Dropdown(
                            choices=list(ui.available_models.keys()),
                            value=ui.current_model_name,
                            label="AI Model Selection",
                            info="Choose which AI model to use for scene analysis (CLIP or BLIP-2)"
                        )

                        scene_model_switch_btn = gr.Button("🔄 Switch Model", variant="secondary")
                        scene_model_status = gr.Markdown(f"✅ Current: {ui.current_model_name}")

                        gr.Markdown("### 🎭 Scene Analysis")
                        scene_image = gr.Image(
                            type="pil",
                            label="Upload Scene Image",
                            height=400
                        )

                        confidence_threshold = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            value=0.6,
                            step=0.05,
                            label="Confidence Threshold",
                            info="Minimum confidence for character detection"
                        )

                        analyze_btn = gr.Button("🔍 Analyze Scene", variant="primary", size="lg")

                        gr.Markdown("""
                        ### Tips:
                        - **Higher confidence** = fewer but more certain detections
                        - **Lower confidence** = more detections but potentially less accurate
                        - Works best with clear, well-lit images
                        - Processing may take 10-30 seconds depending on image complexity
                        """)

                    with gr.Column(scale=2):
                        analysis_results = gr.Markdown(
                            value="Upload a scene image and click 'Analyze Scene' to identify characters.",
                            label="Analysis Results"
                        )

                        scene_visualization = gr.Image(
                            label="Scene Analysis Visualization",
                            height=500
                        )

                # Event handlers for scene analysis
                def switch_scene_model(model_name):
                    status = ui._switch_model(model_name)
                    return status

                scene_model_switch_btn.click(
                    switch_scene_model,
                    inputs=[scene_model_dropdown],
                    outputs=[scene_model_status]
                )

                analyze_btn.click(
                    ui.analyze_scene,
                    inputs=[scene_image, confidence_threshold],
                    outputs=[analysis_results, scene_visualization]
                )

            # About Tab
            with gr.Tab("ℹ️ About", id="about_tab"):
                gr.Markdown("""
                ### About This System
                
                This Naruto Character Recognition System combines state-of-the-art AI models to provide accurate character identification:
                
                #### 🧠 **Technology Stack**
                - **CLIP (Contrastive Language-Image Pre-training)**: For generating image and text embeddings
                - **SAM (Segment Anything Model)**: For automatic object segmentation  
                - **PyTorch**: Deep learning framework
                - **Gradio**: Web interface framework
                
                #### 🎯 **Features**
                - **Database Query**: Find similar characters by image or text description
                - **Scene Analysis**: Automatically detect and identify characters in complex scenes
                - **High Accuracy**: Trained on hundreds of Naruto character images
                - **Real-time Processing**: Fast inference with GPU acceleration (when available)
                
                #### 📊 **Supported Characters**
                The system can identify the following Naruto characters:
                - **Naruto Uzumaki** - The main protagonist
                - **Sakura Haruno** - Team 7 member and medical ninja
                - **Tsunade** - The Fifth Hokage
                - **Gaara** - Kazekage of the Sand Village
                
                #### 🔧 **How It Works**
                1. **Indexing**: Character images are processed with CLIP to create feature embeddings
                2. **Storage**: Embeddings are stored in an efficient vector database
                3. **Query**: User queries are converted to the same embedding space
                4. **Matching**: Cosine similarity finds the most similar characters
                5. **Segmentation**: SAM segments complex scenes into individual objects
                6. **Recognition**: Each segment is analyzed for character identification
                
                #### 📈 **Performance**
                - **Database Size**: 394 character images across 8 classes
                - **Embedding Dimension**: 512D feature vectors
                - **Average Query Time**: < 1 second
                - **Scene Analysis Time**: 10-30 seconds (depending on complexity)
                
                ---
                
                **Built with ❤️ for the Naruto community**
                """)

    return interface


if __name__ == "__main__":
    # Create and launch the interface
    interface = create_gradio_interface()

    # Launch with configuration - try multiple ports if 7860 is busy
    import socket

    def find_free_port():
        """Find a free port on the system."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

    # Try specific ports first, then find a random free port
    ports_to_try = [7860, 7861, 7862, 7863, 7864, 7865]

    launched = False
    for port in ports_to_try:
        try:
            print(f"🚀 Attempting to launch Naruto Character Recognition System on port {port}...")
            interface.launch(
                server_name="127.0.0.1",
                server_port=port,
                share=False,
                debug=False,  # Disable debug to reduce verbosity
                show_error=True,
                inbrowser=True,
                prevent_thread_lock=False
            )
            launched = True
            break
        except OSError as e:
            if "address already in use" in str(e).lower() or "10048" in str(e) or "Cannot find empty port" in str(e):
                print(f"   Port {port} is busy, trying next port...")
                continue
            else:
                print(f"   Unexpected error on port {port}: {e}")
                continue
        except Exception as e:
            print(f"   Error launching on port {port}: {e}")
            continue

    # If all predefined ports failed, try to find any free port
    if not launched:
        try:
            free_port = find_free_port()
            print(f"🚀 All predefined ports busy, trying free port {free_port}...")
            interface.launch(
                server_name="127.0.0.1",
                server_port=free_port,
                share=False,
                debug=False,
                show_error=True,
                inbrowser=True,
                prevent_thread_lock=False
            )
            launched = True
        except Exception as e:
            print(f"   Error launching on free port {free_port}: {e}")

    if not launched:
        print("\n❌ Could not launch the Gradio interface on any port!")
        print("📋 Troubleshooting suggestions:")
        print("   1. Close other applications using ports 7860-7865")
        print("   2. Restart your terminal/IDE")
        print("   3. Try setting environment variable: set GRADIO_SERVER_PORT=8080")
        print("   4. Check if any antivirus software is blocking the ports")
        print("   5. Try running as administrator if needed")