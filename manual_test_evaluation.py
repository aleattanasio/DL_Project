"""
Manual Test Set Evaluation Script

This script creates a small, manually-labeled test set of 20-50 complex scenes
and evaluates the system's top-1 prediction accuracy for each segmented object.
"""

import os
import json
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import random
from pathlib import Path

# Import our existing pipeline components
from scene_analysis_pipeline import SegmentationModel, SegmentProcessor, SearchLogic
from indexing_pipeline import VectorDatabase, EmbeddingModel


class ManualTestSetCreator:
    """Creates and manages manually labeled test sets for evaluation."""

    def __init__(self, test_images_dir: str, output_dir: str = "manual_test_evaluation", csv_file: str = None):
        """
        Initialize the manual test set creator.

        Args:
            test_images_dir: Directory containing test images
            output_dir: Directory to save evaluation results
            csv_file: Path to CSV file containing specific test images to use
        """
        self.test_images_dir = test_images_dir
        self.output_dir = output_dir
        self.csv_file = csv_file
        self.character_classes = ["Gaara", "Naruto", "Sakura", "Tsunade", "Unlabeled", "Other"]

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

    def load_test_images_from_csv(self) -> List[str]:
        """
        Load test images from CSV file and map them to actual images in test directory.

        Returns:
            List of actual image paths from the test directory
        """
        if not self.csv_file or not os.path.exists(self.csv_file):
            print(f"CSV file not found: {self.csv_file}")
            return []

        # Read CSV file
        df = pd.read_csv(self.csv_file)
        csv_filenames = df['filename'].tolist()

        # Get all available test images
        available_images = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            available_images.extend(Path(self.test_images_dir).glob(ext))

        # Since the CSV contains generic names (image_001.jpg, etc.) and the actual files
        # have different names, we'll map them by taking the first N images from the test directory
        # where N is the number of images specified in the CSV
        selected_images = available_images[:len(csv_filenames)]

        print(f"Loaded {len(csv_filenames)} images from CSV: {self.csv_file}")
        print(f"Mapped to {len(selected_images)} actual images from test directory")

        return [str(img) for img in selected_images]

    def select_complex_scenes(self, num_scenes: int = 30) -> List[str]:
        """
        Select a subset of complex scenes from the test directory.
        If CSV file is provided, use those specific images instead.

        Args:
            num_scenes: Number of scenes to select (20-50)

        Returns:
            List of selected image paths
        """
        # If CSV file is provided, use it to select specific images
        if self.csv_file:
            return self.load_test_images_from_csv()

        # Otherwise, use the original random selection
        test_images = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            test_images.extend(Path(self.test_images_dir).glob(ext))

        # Randomly select scenes (you can modify this to be more strategic)
        selected_images = random.sample(test_images, min(num_scenes, len(test_images)))

        print(f"Selected {len(selected_images)} complex scenes for manual evaluation")
        return [str(img) for img in selected_images]

    def create_manual_annotation_template(self, selected_images: List[str]) -> str:
        """
        Create a template file for manual annotation.

        Args:
            selected_images: List of image paths to annotate

        Returns:
            Path to the annotation template file
        """
        template_path = os.path.join(self.output_dir, "manual_annotations_template.json")

        # Create annotation structure
        annotations = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "total_images": len(selected_images),
                "character_classes": self.character_classes,
                "instructions": "For each segment, provide the ground truth character label"
            },
            "images": {}
        }

        for img_path in selected_images:
            img_name = os.path.basename(img_path)
            annotations["images"][img_name] = {
                "path": img_path,
                "segments": [],  # Will be filled after segmentation
                "manual_labels": [],  # To be filled manually
                "notes": ""
            }

        # Save template
        with open(template_path, 'w') as f:
            json.dump(annotations, f, indent=2)

        print(f"Manual annotation template created: {template_path}")
        return template_path


class AccuracyEvaluator:
    """Evaluates system accuracy on manually labeled test set."""

    def __init__(self, vector_db_path: str, output_dir: str = "manual_test_evaluation"):
        """
        Initialize the accuracy evaluator.

        Args:
            vector_db_path: Path to the vector database
            output_dir: Directory to save evaluation results
        """
        self.output_dir = output_dir
        self.character_classes = ["Gaara", "Naruto", "Sakura", "Tsunade", "Unlabeled", "Other"]

        # Initialize pipeline components
        print("Loading models...")
        self.embedding_model = EmbeddingModel()
        self.vector_db = VectorDatabase()
        self.vector_db.load_from_file(vector_db_path)
        self.segmentation_model = SegmentationModel()
        self.segment_processor = SegmentProcessor(self.embedding_model)
        self.search_logic = SearchLogic(self.vector_db)

        print("Models loaded successfully!")

    def segment_and_predict_scene(self, image_path: str) -> List[Dict]:
        """
        Segment a scene and predict characters for each segment.

        Args:
            image_path: Path to the input image

        Returns:
            List of segment predictions with metadata
        """
        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Generate segments
        masks = self.segmentation_model.generate_masks(image_rgb)

        predictions = []

        for i, mask_info in enumerate(masks):
            try:
                # Process segment
                rgba_image, embedding = self.segment_processor.process_segment(
                    image_rgb, mask_info, crop_objects=True
                )

                if embedding is not None:
                    # Find best match
                    matches = self.search_logic.find_best_match(embedding, top_k=1)

                    if matches:
                        top_prediction = matches[0]
                        predicted_character = top_prediction.get('character', 'Unknown')
                        confidence = top_prediction.get('similarity', 0.0)
                    else:
                        predicted_character = 'Unknown'
                        confidence = 0.0

                    # Store prediction info
                    predictions.append({
                        'segment_id': i,
                        'predicted_character': predicted_character,
                        'confidence': confidence,
                        'bbox': self.segment_processor.get_bounding_box(mask_info['segmentation']),
                        'area': np.sum(mask_info['segmentation']),
                        'stability_score': mask_info.get('stability_score', 0.0)
                    })

            except Exception as e:
                print(f"Error processing segment {i}: {e}")
                continue

        return predictions

    def create_annotation_file_with_predictions(self, selected_images: List[str]) -> str:
        """
        Create annotation file with system predictions for manual labeling.

        Args:
            selected_images: List of image paths

        Returns:
            Path to the annotation file
        """
        annotation_path = os.path.join(self.output_dir, "manual_annotations_with_predictions.json")

        annotations = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "total_images": len(selected_images),
                "character_classes": self.character_classes,
                "instructions": "For each segment, verify/correct the predicted character label in 'manual_label' field"
            },
            "images": {}
        }

        print("Processing images and generating predictions...")

        for img_path in selected_images:
            img_name = os.path.basename(img_path)
            print(f"Processing: {img_name}")

            try:
                # Get predictions for this image
                predictions = self.segment_and_predict_scene(img_path)

                # Prepare segments for manual annotation
                segments = []
                for pred in predictions:
                    segments.append({
                        "segment_id": pred['segment_id'],
                        "predicted_character": pred['predicted_character'],
                        "confidence": pred['confidence'],
                        "bbox": pred['bbox'],
                        "area": pred['area'],
                        "stability_score": pred['stability_score'],
                        "manual_label": "",  # To be filled manually
                        "notes": ""
                    })

                annotations["images"][img_name] = {
                    "path": img_path,
                    "total_segments": len(segments),
                    "segments": segments,
                    "image_notes": ""
                }

            except Exception as e:
                print(f"Error processing image {img_name}: {e}")
                annotations["images"][img_name] = {
                    "path": img_path,
                    "error": str(e),
                    "segments": []
                }

        # Save annotations
        with open(annotation_path, 'w') as f:
            json.dump(annotations, f, indent=2)

        print(f"Annotation file with predictions created: {annotation_path}")
        return annotation_path

    def calculate_accuracy(self, annotations_file: str) -> Dict:
        """
        Calculate accuracy metrics from manually annotated file.

        Args:
            annotations_file: Path to completed manual annotations

        Returns:
            Dictionary with accuracy metrics
        """
        # Load annotations
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)

        total_objects = 0
        correct_predictions = 0
        class_stats = {cls: {"total": 0, "correct": 0} for cls in self.character_classes}

        detailed_results = []

        for img_name, img_data in annotations["images"].items():
            if "error" in img_data:
                continue

            for segment in img_data["segments"]:
                if segment["manual_label"] == "":
                    continue  # Skip unannotated segments

                total_objects += 1
                predicted = segment["predicted_character"]
                actual = segment["manual_label"]

                # Count for overall accuracy
                if predicted == actual:
                    correct_predictions += 1

                # Count for per-class accuracy
                if actual in class_stats:
                    class_stats[actual]["total"] += 1
                    if predicted == actual:
                        class_stats[actual]["correct"] += 1

                # Store detailed result
                detailed_results.append({
                    "image": img_name,
                    "segment_id": segment["segment_id"],
                    "predicted": predicted,
                    "actual": actual,
                    "correct": predicted == actual,
                    "confidence": segment["confidence"],
                    "area": segment["area"]
                })

        # Calculate metrics
        overall_accuracy = correct_predictions / total_objects if total_objects > 0 else 0

        per_class_accuracy = {}
        for cls, stats in class_stats.items():
            if stats["total"] > 0:
                per_class_accuracy[cls] = stats["correct"] / stats["total"]
            else:
                per_class_accuracy[cls] = 0

        results = {
            "overall_accuracy": overall_accuracy,
            "total_objects": total_objects,
            "correct_predictions": correct_predictions,
            "per_class_accuracy": per_class_accuracy,
            "class_statistics": class_stats,
            "detailed_results": detailed_results,
            "evaluation_date": datetime.now().isoformat()
        }

        return results

    def save_evaluation_report(self, results: Dict) -> str:
        """
        Save detailed evaluation report.

        Args:
            results: Results dictionary from calculate_accuracy

        Returns:
            Path to the saved report
        """
        report_path = os.path.join(self.output_dir, "accuracy_evaluation_report.json")

        # Save detailed results
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Create summary report
        summary_path = os.path.join(self.output_dir, "accuracy_summary.txt")

        with open(summary_path, 'w') as f:
            f.write("=== MANUAL TEST SET ACCURACY EVALUATION ===\n\n")
            f.write(f"Evaluation Date: {results['evaluation_date']}\n")
            f.write(f"Total Objects Evaluated: {results['total_objects']}\n")
            f.write(f"Correct Predictions: {results['correct_predictions']}\n")
            f.write(f"Overall Accuracy: {results['overall_accuracy']:.3f} ({results['overall_accuracy']*100:.1f}%)\n\n")

            f.write("=== PER-CLASS ACCURACY ===\n")
            for cls, accuracy in results['per_class_accuracy'].items():
                total = results['class_statistics'][cls]['total']
                f.write(f"{cls}: {accuracy:.3f} ({accuracy*100:.1f}%) - {total} objects\n")

            f.write("\n=== DETAILED STATISTICS ===\n")
            for cls, stats in results['class_statistics'].items():
                if stats['total'] > 0:
                    f.write(f"{cls}: {stats['correct']}/{stats['total']} correct\n")

        print(f"Evaluation report saved: {report_path}")
        print(f"Summary report saved: {summary_path}")

        return report_path

    def create_visualization(self, results: Dict) -> str:
        """
        Create visualization of evaluation results.

        Args:
            results: Results dictionary from calculate_accuracy

        Returns:
            Path to the saved visualization
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Overall accuracy bar chart
        ax1.bar(['Overall Accuracy'], [results['overall_accuracy']], color='skyblue')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Overall System Accuracy')
        ax1.set_ylim(0, 1)

        # Add accuracy text on bar
        ax1.text(0, results['overall_accuracy'] + 0.02,
                f"{results['overall_accuracy']:.3f}\n({results['overall_accuracy']*100:.1f}%)",
                ha='center', va='bottom')

        # Per-class accuracy
        classes = list(results['per_class_accuracy'].keys())
        accuracies = list(results['per_class_accuracy'].values())

        bars = ax2.bar(classes, accuracies, color='lightcoral')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Per-Class Accuracy')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)

        # Add accuracy text on bars
        for bar, acc in zip(bars, accuracies):
            ax2.text(bar.get_x() + bar.get_width()/2, acc + 0.02,
                    f"{acc:.2f}", ha='center', va='bottom')

        plt.tight_layout()

        viz_path = os.path.join(self.output_dir, "accuracy_visualization.png")
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Visualization saved: {viz_path}")
        return viz_path


def main():
    """Main function to run the manual test evaluation."""

    # Configuration
    TEST_IMAGES_DIR = "Anime -Naruto-.v1i.multiclass/test"
    VECTOR_DB_PATH = "naruto_embeddings.pkl"  # Adjust path as needed
    CSV_FILE = "manual_test_evaluation/naruto_test_images.csv"  # Use specific test images from CSV
    NUM_TEST_SCENES = 30  # Number of scenes to evaluate (20-50)

    print("=== MANUAL TEST SET EVALUATION ===")
    print("This script will:")
    print("1. Load specific test images from naruto_test_images.csv")
    print("2. Generate system predictions for segmented objects")
    print("3. Create annotation file for manual labeling")
    print("4. Calculate accuracy after manual annotation")
    print()

    # Step 1: Create test set using CSV file
    print("Step 1: Loading test images from CSV...")
    test_creator = ManualTestSetCreator(TEST_IMAGES_DIR, csv_file=CSV_FILE)
    selected_images = test_creator.select_complex_scenes(NUM_TEST_SCENES)

    # Step 2: Generate predictions and create annotation file
    print("Step 2: Loading models and generating predictions...")
    evaluator = AccuracyEvaluator(VECTOR_DB_PATH)
    annotation_file = evaluator.create_annotation_file_with_predictions(selected_images)

    print("\n" + "="*50)
    print("MANUAL ANNOTATION REQUIRED")
    print("="*50)
    print(f"Please manually annotate the file: {annotation_file}")
    print("For each segment, fill in the 'manual_label' field with the correct character name.")
    print("Available classes:", evaluator.character_classes)
    print("Once completed, run this script again with --evaluate flag")
    print("="*50)

    # Note: In a full implementation, you might want to add command line arguments
    # to handle the evaluation step separately after manual annotation


def evaluate_completed_annotations(annotation_file: str, vector_db_path: str):
    """
    Evaluate accuracy from completed manual annotations.

    Args:
        annotation_file: Path to completed manual annotations
        vector_db_path: Path to vector database
    """
    print("=== EVALUATING COMPLETED ANNOTATIONS ===")

    evaluator = AccuracyEvaluator(vector_db_path)

    # Calculate accuracy
    results = evaluator.calculate_accuracy(annotation_file)

    # Save reports
    report_path = evaluator.save_evaluation_report(results)
    viz_path = evaluator.create_visualization(results)

    # Print summary
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Total Objects Evaluated: {results['total_objects']}")
    print(f"Overall Accuracy: {results['overall_accuracy']:.3f} ({results['overall_accuracy']*100:.1f}%)")
    print("\nPer-Class Accuracy:")
    for cls, accuracy in results['per_class_accuracy'].items():
        total = results['class_statistics'][cls]['total']
        if total > 0:
            print(f"  {cls}: {accuracy:.3f} ({accuracy*100:.1f}%) - {total} objects")

    print(f"\nDetailed report: {report_path}")
    print(f"Visualization: {viz_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--evaluate":
        # Evaluate mode - calculate accuracy from completed annotations
        annotation_file = "manual_test_evaluation/manual_annotations_with_predictions.json"
        vector_db_path = "naruto_embeddings.pkl"

        if os.path.exists(annotation_file):
            evaluate_completed_annotations(annotation_file, vector_db_path)
        else:
            print(f"Annotation file not found: {annotation_file}")
            print("Please run the script without --evaluate flag first to create the test set")
    else:
        # Create test set mode
        main()

