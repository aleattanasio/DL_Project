import sys
import pandas as pd
import json
import pickle
from pathlib import Path
from collections import defaultdict

sys.path.append('..')
from scene_analysis_pipeline import SceneAnalysisPipeline
from indexing_pipeline import VectorDatabase
import clip
import torch


def load_finetuned_clip_model(model_name, checkpoint_path, device):
    # Load CLIP and apply fine-tuned weights (if available). Returns (model, preprocess).
    model, preprocess = clip.load(model_name, device=device)
    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'], strict=False)
            print(f"✅ Fine-tuned weights loaded from: {checkpoint_path}")
        else:
            print("⚠️ Checkpoint missing 'model_state_dict', using base model.")
    else:
        print("⚠️ Checkpoint not found, using base model.")
    model.eval()
    return model, preprocess


def load_finetuned_embeddings_db(finetuned_db_path):
    finetuned_db_path = Path(finetuned_db_path)
    if not finetuned_db_path.exists():
        raise FileNotFoundError(f"File not found: {finetuned_db_path}")
    with open(finetuned_db_path, 'rb') as f:
        raw = pickle.load(f)
    vector_db = VectorDatabase()
    if isinstance(raw, dict) and 'embeddings_matrix' in raw and 'data' in raw:
        embeddings = raw['embeddings_matrix']
        entries = raw['data']
        if len(entries) != embeddings.shape[0]:
            raise ValueError("Mismatch between number of entries and embeddings_matrix rows")
        for i, entry in enumerate(entries):
            label = entry.get('label') or entry.get('class') or 'Unknown'
            image_path = entry.get('image_path') or entry.get('filename') or f"sample_{i}.png"
            emb = embeddings[i]
            vector_db.add_entry(
                image_path=image_path,
                label=label,
                embedding=emb,
                metadata={k: v for k, v in entry.items() if k not in ['label', 'image_path', 'filename']}
            )
        print(f"✅ Fine-tuned database (matrix+data) loaded: {len(vector_db.get_labels())} embeddings")
        return vector_db
    if isinstance(raw, dict) and 'data' in raw and isinstance(raw['data'], list) and 'next_id' in raw:
        for entry in raw['data']:
            vector_db.add_entry(
                image_path=entry['image_path'],
                label=entry['label'],
                embedding=entry['embedding'],
                metadata=entry.get('metadata', {})
            )
        print(f"✅ Fine-tuned database (raw data) loaded: {len(vector_db.get_labels())} embeddings")
        return vector_db
    raise ValueError("Unexpected fine-tuned database format.")


class ManualTestEvaluator:
    # Evaluator for manual test set using the scene analysis pipeline

    def __init__(self,
                 vector_db_path,
                 sam_model_type="vit_h",
                 confidence_threshold=0.0,
                 use_finetuned_db=True,
                 finetuned_model_checkpoint=None,
                 finetuned_clip_model_name="ViT-B/32",
                 device=None):
        print("Initializing evaluator with Scene Analysis Pipeline...")
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if use_finetuned_db:
            print(f"Loading fine-tuned database: {vector_db_path}")
            vector_db = load_finetuned_embeddings_db(vector_db_path)
        else:
            print(f"Loading pre-trained database: {vector_db_path}")
            vector_db = VectorDatabase()
            vector_db.load_from_file(vector_db_path)
        print(f"Database loaded: {len(vector_db.get_labels())} images")
        print(f"Available classes: {set(vector_db.get_labels())}")
        finetuned_clip_model = None
        finetuned_preprocess = None
        if finetuned_model_checkpoint:
            try:
                finetuned_clip_model, finetuned_preprocess = load_finetuned_clip_model(
                    finetuned_clip_model_name,
                    finetuned_model_checkpoint,
                    device
                )
            except Exception as e:
                print(f"⚠️ Error loading fine-tuned model: {e}")
        self.pipeline = SceneAnalysisPipeline(
            vector_database=vector_db,
            sam_model_type=sam_model_type,
            clip_model=finetuned_clip_model,
            clip_preprocess=finetuned_preprocess,
            clip_model_name=finetuned_clip_model_name
        )
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


def evaluate_test_set(csv_file,
                      labels_file,
                      vector_db_path,
                      results_file,
                      sam_model_type="vit_h",
                      save_visualizations=False,
                      use_finetuned_db=True,
                      finetuned_model_checkpoint=None,
                      finetuned_clip_model_name="ViT-B/32"):
    df = pd.read_csv(csv_file)
    manual_labels = load_manual_labels(Path(labels_file))
    if manual_labels is None:
        return
    print(f"\nUsing SAM model: {sam_model_type}")
    evaluator = ManualTestEvaluator(
        vector_db_path,
        sam_model_type=sam_model_type,
        use_finetuned_db=use_finetuned_db,
        finetuned_model_checkpoint=finetuned_model_checkpoint,
        finetuned_clip_model_name=finetuned_clip_model_name
    )
    results = []
    correct = 0
    total = 0
    class_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    print("\n" + "=" * 80)
    print("MANUAL TEST SET EVALUATION - SCENE ANALYSIS PIPELINE (Fine-Tuned)")
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
    print("FINAL RESULTS (Fine-Tuned DB)")
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
            'confidence_threshold': 0.0,
            'vector_database': str(vector_db_path),
            'fine_tuned': use_finetuned_db,
            'finetuned_checkpoint': finetuned_model_checkpoint
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


def load_manual_labels(labels_file):
    if not Path(labels_file).exists():
        print(f"\nERROR: {labels_file} not found!")
        return None
    with open(labels_file, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    TEST_DIR = Path("manual_test_evaluation")
    CSV_FILE = TEST_DIR / "naruto_test_images.csv"
    LABELS_FILE = TEST_DIR / "manual_labels.json"
    RESULTS_FILE = TEST_DIR / "evaluation_results_finetuned.json"
    VECTOR_DB_PATH = "results_clip_finetuned/databases/naruto_finetuned_embeddings.pkl"
    FINETUNED_CHECKPOINT = "results_clip_finetuned/checkpoints/best_model.pth"
    if not LABELS_FILE.exists():
        print("Manual labels file missing.")
        sys.exit(1)

    evaluate_test_set(
        csv_file=CSV_FILE,
        labels_file=LABELS_FILE,
        vector_db_path=VECTOR_DB_PATH,
        results_file=RESULTS_FILE,
        sam_model_type="vit_h",
        save_visualizations=False,
        use_finetuned_db=True,
        finetuned_model_checkpoint=FINETUNED_CHECKPOINT,
        finetuned_clip_model_name="ViT-B/32"
    )