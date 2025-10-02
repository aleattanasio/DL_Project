import sys
import pandas as pd
import json
from pathlib import Path
from collections import defaultdict

sys.path.append('..')
from scene_analysis_pipeline import SceneAnalysisPipeline
from indexing_pipeline import VectorDatabase


# Evaluates the manual test set using Scene Analysis Pipeline.
class ManualTestEvaluator:

    def __init__(self, vector_db_path, sam_model_type="vit_h", confidence_threshold=0.0):
        print("Initializing evaluator with Scene Analysis Pipeline...")

        vector_db = VectorDatabase()
        vector_db.load_from_file(vector_db_path)

        print(f"Database loaded: {len(vector_db.get_labels())} images")
        print(f"Available classes: {set(vector_db.get_labels())}")

        self.pipeline = SceneAnalysisPipeline(vector_db, sam_model_type=sam_model_type)
        self.confidence_threshold = confidence_threshold

    def analyze_image(self, image_path):
        try:
            results = self.pipeline.analyze_scene(
                str(image_path),
                confidence_threshold=self.confidence_threshold,
                save_visualizations=False
            )
            return results
        except Exception as e:
            print(f"Error analyzing {image_path}: {e}")
            return None

    def get_top1_prediction(self, image_path):
        results = self.analyze_image(image_path)

        if results is None or not results.get('all_predictions'):
            return None

        best_prediction = None
        best_confidence = -1

        for pred in results['all_predictions']:
            if pred['prediction'] != 'Unknown' and pred['confidence'] > best_confidence:
                best_prediction = pred['prediction']
                best_confidence = pred['confidence']

        return best_prediction if best_prediction else 'Unknown'

    def get_all_predictions(self, image_path):
        results = self.analyze_image(image_path)

        if results is None:
            return []

        predictions = []
        for pred in results.get('all_predictions', []):
            if pred['prediction'] != 'Unknown':
                predictions.append((pred['prediction'], pred['confidence']))

        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions


def load_manual_labels(labels_file):
    if not labels_file.exists():
        print(f"\nERROR: {labels_file} not found!")
        print("\nCreate the file with this format:")
        print('{\n  "image_001.jpg": "naruto",\n  "image_002.jpg": "sasuke",\n  ...\n}')
        return None

    with open(labels_file, 'r') as f:
        return json.load(f)


def create_labels_template(csv_file, output_file):
    df = pd.read_csv(csv_file)
    template = {}

    for _, row in df.iterrows():
        filename = row['filename'].strip()
        if filename:
            template[filename] = "INSERT_CLASS_NAME_HERE"

    with open(output_file, 'w') as f:
        json.dump(template, f, indent=2)

    print(f"✅ Template created: {output_file}")
    print("  Fill the file with the correct label for each image.")


def evaluate_test_set(csv_file, labels_file, vector_db_path, results_file,
                      sam_model_type="vit_h", save_visualizations=False):
    df = pd.read_csv(csv_file)
    manual_labels = load_manual_labels(labels_file)

    if manual_labels is None:
        return

    print(f"\nUsing SAM model: {sam_model_type}")
    evaluator = ManualTestEvaluator(vector_db_path, sam_model_type=sam_model_type)

    results = []
    correct = 0
    total = 0
    class_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

    print("\n" + "=" * 80)
    print("MANUAL TEST SET EVALUATION - SCENE ANALYSIS PIPELINE")
    print("=" * 80)

    test_dir = Path(csv_file).parent

    if save_visualizations:
        evaluator.pipeline._save_visualizations = True

    for idx, row in df.iterrows():
        filename = row['filename'].strip()
        if not filename:
            continue

        image_path = test_dir / filename

        if not image_path.exists():
            print(f"Warning: {filename}: file not found, skipping")
            continue

        if filename not in manual_labels:
            print(f"Warning: {filename}: manual label missing, skipping")
            continue

        ground_truth = manual_labels[filename]

        print(f"\nProcessing {filename}...")

        prediction = evaluator.get_top1_prediction(image_path)

        if prediction is None:
            print(f"Warning: {filename}: prediction error, skipping")
            continue

        all_predictions = evaluator.get_all_predictions(image_path)

        is_correct = (prediction == ground_truth)
        if is_correct:
            correct += 1
        total += 1

        class_stats[ground_truth]['total'] += 1
        if is_correct:
            class_stats[ground_truth]['correct'] += 1

        result = {
            'filename': filename,
            'ground_truth': ground_truth,
            'top1_prediction': prediction,
            'correct': is_correct,
            'all_predictions': [
                {'class': pred[0], 'confidence': float(pred[1])}
                for pred in all_predictions[:5]
            ]
        }
        results.append(result)

        status = "✅" if is_correct else "❌"
        print(f"{status} {filename:20s} | GT: {ground_truth:15s} | PRED: {prediction:15s}")
        if all_predictions:
            top_preds = ', '.join([f"{cls}({conf:.3f})" for cls, conf in all_predictions[:3]])
            print(f"    Top-3: {top_preds}")

    accuracy = (correct / total * 100) if total > 0 else 0

    per_class_accuracy = {}
    for cls, stats in class_stats.items():
        cls_acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        per_class_accuracy[cls] = {
            'accuracy': cls_acc,
            'correct': stats['correct'],
            'total': stats['total']
        }

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Total tested images: {total}")
    print(f"Correct top-1 predictions: {correct}")
    print(f"OVERALL ACCURACY: {accuracy:.2f}%")
    print("\n" + "-" * 80)
    print("PER-CLASS ACCURACY:")
    print("-" * 80)

    for cls in sorted(per_class_accuracy.keys()):
        stats = per_class_accuracy[cls]
        print(f"{cls:20s}: {stats['accuracy']:6.2f}% ({stats['correct']}/{stats['total']})")

    print("=" * 80)

    output = {
        'evaluation_info': {
            'sam_model': sam_model_type,
            'confidence_threshold': evaluator.confidence_threshold,
            'vector_database': str(vector_db_path)
        },
        'total_samples': total,
        'correct_predictions': correct,
        'overall_accuracy': accuracy,
        'per_class_accuracy': per_class_accuracy,
        'detailed_results': results
    }

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Results saved to: {results_file}")

    return accuracy, results


if __name__ == "__main__":
    TEST_DIR = Path("manual_test_evaluation")
    CSV_FILE = TEST_DIR / "naruto_test_images.csv"
    LABELS_FILE = TEST_DIR / "manual_labels.json"
    RESULTS_FILE = TEST_DIR / "evaluation_results.json"
    VECTOR_DB_PATH = "naruto_embeddings.pkl"

    if not LABELS_FILE.exists():
        print("Manual labels file not found.")
        response = input("Do you want to create a template? (y/n): ")
        if response.lower() == 'y':
            template_file = TEST_DIR / "manual_labels_template.json"
            create_labels_template(CSV_FILE, template_file)
            print(f"\nFill {template_file} and rename it to 'manual_labels.json'")
        sys.exit(0)

    try:
        evaluate_test_set(
            csv_file=CSV_FILE,
            labels_file=LABELS_FILE,
            vector_db_path=VECTOR_DB_PATH,
            results_file=RESULTS_FILE,
            sam_model_type="vit_h",
            save_visualizations=False
        )
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
