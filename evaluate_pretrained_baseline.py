# Evaluate Pre-trained CLIP Model as Baseline for Zero-Shot Performance
# This script evaluates the original CLIP model without any fine-tuning to establish a baseline.

import torch
import clip
import numpy as np
import pandas as pd
from PIL import Image
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from torch.utils.data import Dataset, DataLoader

# Dataset for loading ALL images (train + valid + test) for comprehensive evaluation
class NarutoFullDataset(Dataset):

    def __init__(self, data_dir, preprocess_fn, splits=['train', 'valid', 'test']):
        self.data_dir = data_dir
        self.preprocess = preprocess_fn
        self.splits = splits
        self.samples = self._load_all_samples()
        self.class_to_idx = self._create_class_mapping()
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        print(f"Loaded {len(self.samples)} samples from splits: {splits}")
        print(f"Classes: {list(self.class_to_idx.keys())}")

        split_counts = self._get_split_statistics()
        for split, count in split_counts.items():
            print(f"  {split}: {count} samples")

    def _load_all_samples(self):
        all_samples = []
        for split in self.splits:
            csv_path = os.path.join(self.data_dir, split, '_classes.csv')
            if not os.path.exists(csv_path):
                print(f"Warning: {csv_path} not found, skipping {split} split")
                continue
            df = pd.read_csv(csv_path)
            class_columns = [col for col in df.columns if col != 'filename']
            for _, row in df.iterrows():
                filename = row['filename']
                image_path = os.path.join(self.data_dir, split, filename)
                if not os.path.exists(image_path):
                    continue
                active_classes = [col for col in class_columns if row[col] == 1]
                if len(active_classes) == 1:
                    label = active_classes[0]
                    all_samples.append({
                        'image_path': image_path,
                        'label': label,
                        'filename': filename,
                        'split': split
                    })
        return all_samples

    def _create_class_mapping(self):
        unique_labels = sorted(set(sample['label'] for sample in self.samples))
        return {label: idx for idx, label in enumerate(unique_labels)}

    def _get_split_statistics(self):
        split_counts = {}
        for sample in self.samples:
            split = sample['split']
            split_counts[split] = split_counts.get(split, 0) + 1
        return split_counts

    def get_class_distribution_by_split(self):
        distribution = {}
        for split in self.splits:
            distribution[split] = {}
        for sample in self.samples:
            split = sample['split']
            label = sample['label']
            if label not in distribution[split]:
                distribution[split][label] = 0
            distribution[split][label] += 1
        return distribution

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert('RGB')
        image = self.preprocess(image)
        label_idx = self.class_to_idx[sample['label']]
        filename = sample['filename']
        split = sample['split']
        return image, label_idx, filename, split


# Simple dataset for loading test images without any preprocessing changes
class NarutoTestDataset(Dataset):

    def __init__(self, data_dir, preprocess_fn):
        self.data_dir = data_dir
        self.preprocess = preprocess_fn
        self.samples = self._load_samples()
        self.class_to_idx = self._create_class_mapping()
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        print(f"Loaded {len(self.samples)} test samples")
        print(f"Classes: {list(self.class_to_idx.keys())}")

    def _load_samples(self):
        csv_path = os.path.join(self.data_dir, 'test', '_classes.csv')
        df = pd.read_csv(csv_path)
        samples = []
        class_columns = [col for col in df.columns if col != 'filename']
        for _, row in df.iterrows():
            filename = row['filename']
            image_path = os.path.join(self.data_dir, 'test', filename)
            if not os.path.exists(image_path):
                continue
            active_classes = [col for col in class_columns if row[col] == 1]
            if len(active_classes) == 1:
                label = active_classes[0]
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
        image = self.preprocess(image)
        label_idx = self.class_to_idx[sample['label']]
        filename = sample['filename']
        return image, label_idx, filename


# Evaluator for pre-trained CLIP model (zero-shot baseline)
class PretrainedCLIPEvaluator:

    def __init__(self, model_name="ViT-B/32", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        print(f"Loading pre-trained CLIP model: {model_name}")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        print(f"✅ Pre-trained CLIP model loaded on {self.device}")
        print(f"Model: {model_name}")
        print(f"Zero-shot evaluation mode")

    def create_text_prompts(self, class_names):
        templates = [
            "a photo of {}",
            "a picture of {}",
            "an image of {}",
            "{} from Naruto anime",
            "the character {}",
        ]
        prompts = []
        for class_name in class_names:
            class_prompts = [template.format(class_name) for template in templates]
            prompts.extend(class_prompts)
        return prompts

    def get_text_embeddings(self, class_names):
        prompts = self.create_text_prompts(class_names)
        text_tokens = clip.tokenize(prompts).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        templates_per_class = 5
        num_classes = len(class_names)
        averaged_embeddings = []
        for i in range(num_classes):
            start_idx = i * templates_per_class
            end_idx = start_idx + templates_per_class
            class_embeddings = text_features[start_idx:end_idx]
            averaged_embedding = class_embeddings.mean(dim=0)
            averaged_embeddings.append(averaged_embedding)
        return torch.stack(averaged_embeddings)

    def evaluate_zero_shot(self, data_dir, save_dir="results_pretrained_baseline"):
        print("\nEVALUATING PRE-TRAINED CLIP MODEL (ZERO-SHOT)")
        print("=" * 60)
        test_dataset = NarutoTestDataset(data_dir, self.preprocess)
        test_loader = DataLoader(
            test_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        class_names = list(test_dataset.class_to_idx.keys())
        class_to_idx = test_dataset.class_to_idx
        idx_to_class = test_dataset.idx_to_class
        print(f"Test set size: {len(test_dataset)} samples")
        print(f"Classes: {class_names}")
        print("Creating text embeddings for zero-shot classification...")
        text_embeddings = self.get_text_embeddings(class_names)
        all_predictions = []
        all_labels = []
        all_filenames = []
        all_confidences = []
        per_sample_results = []
        print("Performing zero-shot evaluation...")
        with torch.no_grad():
            for batch_idx, (images, labels, filenames) in enumerate(tqdm(test_loader, desc="Zero-shot Evaluation")):
                images = images.to(self.device)
                labels = labels.to(self.device)
                image_features = self.model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                similarities = torch.matmul(image_features, text_embeddings.t())
                confidences = torch.softmax(similarities * 100, dim=1)
                predictions = torch.argmax(similarities, dim=1)
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
                        'zero_shot_prediction': pred_class
                    })
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_confidences = np.array(all_confidences)
        overall_accuracy = accuracy_score(all_labels, all_predictions)
        correct_predictions = np.sum(all_predictions == all_labels)
        total_predictions = len(all_labels)
        print(f"\nZERO-SHOT EVALUATION RESULTS")
        print("=" * 50)
        print(f"Overall Zero-Shot Accuracy: {overall_accuracy:.4f} ({correct_predictions}/{total_predictions})")
        print(f"Percentage of objects correctly identified: {overall_accuracy * 100:.2f}%")
        class_report = classification_report(all_labels, all_predictions,
                                             target_names=class_names,
                                             output_dict=True)
        print(f"\nPER-CLASS ZERO-SHOT PERFORMANCE:")
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
        cm = confusion_matrix(all_labels, all_predictions)
        os.makedirs(os.path.join(save_dir, "reports"), exist_ok=True)
        evaluation_results = {
            'evaluation_type': 'zero_shot_baseline',
            'overall_metrics': {
                'zero_shot_accuracy': float(overall_accuracy),
                'correct_predictions': int(correct_predictions),
                'total_predictions': int(total_predictions),
                'percentage_correct': float(overall_accuracy * 100)
            },
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': cm.tolist(),
            'class_to_idx': class_to_idx,
            'classification_report': class_report,
            'model_info': {
                'model_name': self.model_name,
                'evaluation_type': 'zero_shot',
                'text_prompts_used': self.create_text_prompts(class_names[:1]),
                'device': str(self.device)
            }
        }
        report_path = os.path.join(save_dir, "reports", "zero_shot_evaluation_report.json")
        with open(report_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        print(f"Zero-shot evaluation report saved to: {report_path}")
        predictions_path = os.path.join(save_dir, "reports", "zero_shot_per_sample_predictions.json")
        with open(predictions_path, 'w') as f:
            json.dump(per_sample_results, f, indent=2)
        print(f"Per-sample predictions saved to: {predictions_path}")
        self._plot_confusion_matrix(cm, class_names,
                                    os.path.join(save_dir, "plots", "zero_shot_confusion_matrix.png"))
        self._save_evaluation_summary(evaluation_results, per_sample_results,
                                      os.path.join(save_dir, "reports", "zero_shot_evaluation_summary.md"))
        print(f"Confusion matrix plot saved to: {os.path.join(save_dir, 'plots', 'zero_shot_confusion_matrix.png')}")
        print(f"Evaluation summary saved to: {os.path.join(save_dir, 'reports', 'zero_shot_evaluation_summary.md')}")
        return evaluation_results

    def _plot_confusion_matrix(self, cm, class_names, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Zero-Shot Test Set Confusion Matrix\n(Pre-trained CLIP)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _save_evaluation_summary(self, eval_results, per_sample_results, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("# Zero-Shot Evaluation Summary (Pre-trained CLIP)\n\n")
            f.write("## Overall Performance\n\n")
            overall = eval_results['overall_metrics']
            f.write(f"- **Zero-Shot Accuracy**: {overall['zero_shot_accuracy']:.4f} ({overall['percentage_correct']:.2f}%)\n")
            f.write(f"- **Correct Predictions**: {overall['correct_predictions']}/{overall['total_predictions']}\n")
            f.write(f"- **Total Test Samples**: {overall['total_predictions']}\n")
            f.write(f"- **Evaluation Type**: Zero-shot (no fine-tuning)\n\n")
            f.write("## Per-Class Performance\n\n")
            f.write("| Class | Accuracy | Correct/Total | Precision | Recall | F1-Score |\n")
            f.write("|-------|----------|---------------|-----------|--------|----------|\n")
            for class_name, metrics in eval_results['per_class_metrics'].items():
                f.write(f"| {class_name} | {metrics['accuracy']:.4f} | "
                        f"{metrics['correct']}/{metrics['total']} | "
                        f"{metrics['precision']:.3f} | {metrics['recall']:.3f} | "
                        f"{metrics['f1_score']:.3f} |\n")
            f.write("\n## Model Configuration\n\n")
            model_info = eval_results['model_info']
            for key, value in model_info.items():
                f.write(f"- **{key.replace('_', ' ').title()}**: {value}\n")
            f.write("\n## Sample Predictions (First 15)\n\n")
            f.write("| Filename | True Class | Predicted Class | Confidence | Correct |\n")
            f.write("|----------|------------|-----------------|------------|----------|\n")
            for i, sample in enumerate(per_sample_results[:15]):
                correct_emoji = "✅" if sample['correct'] else "❌"
                f.write(f"| {sample['filename']} | {sample['true_class']} | "
                        f"{sample['predicted_class']} | {sample['confidence']:.3f} | {correct_emoji} |\n")
            if len(per_sample_results) > 15:
                f.write(f"\n... and {len(per_sample_results) - 15} more samples\n")
            f.write("\n## Error Analysis\n\n")
            incorrect_samples = [s for s in per_sample_results if not s['correct']]
            if incorrect_samples:
                f.write(f"**Total Errors**: {len(incorrect_samples)}\n\n")
                error_by_class = {}
                for sample in incorrect_samples:
                    true_class = sample['true_class']
                    pred_class = sample['predicted_class']
                    if true_class not in error_by_class:
                        error_by_class[true_class] = {}
                    if pred_class not in error_by_class[true_class]:
                        error_by_class[true_class][pred_class] = 0
                    error_by_class[true_class][pred_class] += 1
                f.write("**Common Misclassifications**:\n\n")
                for true_class, predictions in error_by_class.items():
                    f.write(f"- **{true_class}** confused with:\n")
                    for pred_class, count in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
                        f.write(f"  - {pred_class}: {count} times\n")
                    f.write("\n")
            else:
                f.write("Perfect zero-shot classification! No errors found.\n")
            f.write("\n## Methodology\n\n")
            f.write("This evaluation uses zero-shot classification with pre-trained CLIP:\n")
            f.write("- No fine-tuning on Naruto character data\n")
            f.write("- Uses multiple text prompt templates for robust classification\n")
            f.write("- Averages embeddings across templates for each class\n")
            f.write("- Represents baseline performance before fine-tuning\n")

    def evaluate_zero_shot_full_dataset(self, data_dir, save_dir="results_pretrained_baseline"):
        print("\nEVALUATING PRE-TRAINED CLIP MODEL ON FULL DATASET (ZERO-SHOT)")
        print("=" * 70)
        print("This evaluation includes ALL samples: train + validation + test")
        full_dataset = NarutoFullDataset(data_dir, self.preprocess, ['train', 'valid', 'test'])
        full_loader = DataLoader(
            full_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        class_names = list(full_dataset.class_to_idx.keys())
        class_to_idx = full_dataset.class_to_idx
        idx_to_class = full_dataset.idx_to_class
        print(f"Full dataset size: {len(full_dataset)} samples")
        print(f"Classes: {class_names}")
        split_distribution = full_dataset.get_class_distribution_by_split()
        print(f"\nDataset Distribution by Split:")
        for split, class_dist in split_distribution.items():
            total_split = sum(class_dist.values()) if class_dist else 0
            print(f"  {split}: {total_split} samples")
            for class_name, count in sorted(class_dist.items(), key=lambda x: x[1], reverse=True):
                print(f"    {class_name}: {count}")
        print("\nCreating text embeddings for zero-shot classification...")
        text_embeddings = self.get_text_embeddings(class_names)
        all_predictions = []
        all_labels = []
        all_filenames = []
        all_splits = []
        all_confidences = []
        per_sample_results = []
        print("Performing zero-shot evaluation on full dataset...")
        with torch.no_grad():
            for batch_idx, (images, labels, filenames, splits) in enumerate(tqdm(full_loader, desc="Full Dataset Zero-shot")):
                images = images.to(self.device)
                labels = labels.to(self.device)
                image_features = self.model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                similarities = torch.matmul(image_features, text_embeddings.t())
                confidences = torch.softmax(similarities * 100, dim=1)
                predictions = torch.argmax(similarities, dim=1)
                batch_predictions = predictions.cpu().numpy()
                batch_labels = labels.cpu().numpy()
                batch_confidences = confidences.cpu().numpy()
                all_predictions.extend(batch_predictions)
                all_labels.extend(batch_labels)
                all_filenames.extend(filenames)
                all_splits.extend(splits)
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
                        'zero_shot_prediction': pred_class,
                        'split': splits[i]
                    })
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_confidences = np.array(all_confidences)
        all_splits = np.array(all_splits)
        overall_accuracy = accuracy_score(all_labels, all_predictions)
        correct_predictions = np.sum(all_predictions == all_labels)
        total_predictions = len(all_labels)
        print(f"\nFULL DATASET ZERO-SHOT EVALUATION RESULTS")
        print("=" * 60)
        print(f"Overall Zero-Shot Accuracy: {overall_accuracy:.4f} ({correct_predictions}/{total_predictions})")
        print(f"Percentage of objects correctly identified: {overall_accuracy * 100:.2f}%")
        split_accuracies = {}
        print(f"\nACCURACY BY SPLIT:")
        print("-" * 30)
        for split in ['train', 'valid', 'test']:
            split_mask = all_splits == split
            if np.any(split_mask):
                split_preds = all_predictions[split_mask]
                split_labels = all_labels[split_mask]
                split_accuracy = accuracy_score(split_labels, split_preds)
                split_correct = np.sum(split_preds == split_labels)
                split_total = len(split_labels)
                split_accuracies[split] = {
                    'accuracy': float(split_accuracy),
                    'correct': int(split_correct),
                    'total': int(split_total)
                }
                print(f"  {split:>5}: {split_accuracy:.4f} ({split_correct}/{split_total}) - {split_accuracy*100:.2f}%")
        class_report = classification_report(all_labels, all_predictions,
                                             target_names=class_names,
                                             output_dict=True)
        print(f"\nPER-CLASS ZERO-SHOT PERFORMANCE (Full Dataset):")
        print("-" * 50)
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
        cm = confusion_matrix(all_labels, all_predictions)
        os.makedirs(os.path.join(save_dir, "reports"), exist_ok=True)
        evaluation_results = {
            'evaluation_type': 'zero_shot_full_dataset',
            'overall_metrics': {
                'zero_shot_accuracy': float(overall_accuracy),
                'correct_predictions': int(correct_predictions),
                'total_predictions': int(total_predictions),
                'percentage_correct': float(overall_accuracy * 100)
            },
            'split_accuracies': split_accuracies,
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': cm.tolist(),
            'class_to_idx': class_to_idx,
            'classification_report': class_report,
            'dataset_distribution': split_distribution,
            'model_info': {
                'model_name': self.model_name,
                'evaluation_type': 'zero_shot_full_dataset',
                'splits_included': ['train', 'valid', 'test'],
                'text_prompts_used': self.create_text_prompts(class_names[:1]),
                'device': str(self.device)
            }
        }
        report_path = os.path.join(save_dir, "reports", "zero_shot_full_dataset_report.json")
        with open(report_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        print(f"\nFull dataset evaluation report saved to: {report_path}")
        predictions_path = os.path.join(save_dir, "reports", "zero_shot_full_dataset_predictions.json")
        with open(predictions_path, 'w') as f:
            json.dump(per_sample_results, f, indent=2)
        print(f"Per-sample predictions saved to: {predictions_path}")
        self._plot_confusion_matrix_full(cm, class_names,
                                         os.path.join(save_dir, "plots", "zero_shot_full_dataset_confusion_matrix.png"))
        self._save_evaluation_summary_full(evaluation_results, per_sample_results,
                                           os.path.join(save_dir, "reports", "zero_shot_full_dataset_summary.md"))
        print(f"Confusion matrix plot saved to: {os.path.join(save_dir, 'plots', 'zero_shot_full_dataset_confusion_matrix.png')}")
        print(f"Evaluation summary saved to: {os.path.join(save_dir, 'reports', 'zero_shot_full_dataset_summary.md')}")
        return evaluation_results

    def _plot_confusion_matrix_full(self, cm, class_names, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Zero-Shot Full Dataset Confusion Matrix\n(Pre-trained CLIP - Train + Valid + Test)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _save_evaluation_summary_full(self, eval_results, per_sample_results, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("# Zero-Shot Full Dataset Evaluation Summary (Pre-trained CLIP)\n\n")
            f.write("## Overall Performance (Train + Valid + Test)\n\n")
            overall = eval_results['overall_metrics']
            f.write(f"- **Zero-Shot Accuracy**: {overall['zero_shot_accuracy']:.4f} ({overall['percentage_correct']:.2f}%)\n")
            f.write(f"- **Correct Predictions**: {overall['correct_predictions']}/{overall['total_predictions']}\n")
            f.write(f"- **Total Samples**: {overall['total_predictions']}\n")
            f.write(f"- **Evaluation Type**: Zero-shot (no fine-tuning)\n")
            f.write(f"- **Dataset Coverage**: Full dataset (all splits)\n\n")
            f.write("## Performance by Split\n\n")
            f.write("| Split | Accuracy | Correct/Total | Percentage |\n")
            f.write("|-------|----------|---------------|------------|\n")
            for split, metrics in eval_results['split_accuracies'].items():
                f.write(f"| {split.title()} | {metrics['accuracy']:.4f} | "
                        f"{metrics['correct']}/{metrics['total']} | "
                        f"{metrics['accuracy']*100:.2f}% |\n")
            f.write("\n## Per-Class Performance (Full Dataset)\n\n")
            f.write("| Class | Accuracy | Correct/Total | Precision | Recall | F1-Score |\n")
            f.write("|-------|----------|---------------|-----------|--------|----------|\n")
            for class_name, metrics in eval_results['per_class_metrics'].items():
                f.write(f"| {class_name} | {metrics['accuracy']:.4f} | "
                        f"{metrics['correct']}/{metrics['total']} | "
                        f"{metrics['precision']:.3f} | {metrics['recall']:.3f} | "
                        f"{metrics['f1_score']:.3f} |\n")
            f.write("\n## Dataset Distribution\n\n")
            distribution = eval_results['dataset_distribution']
            f.write("| Split | Total | Gara | Naruto | Sakura | Tsunade |\n")
            f.write("|-------|-------|------|--------|--------|----------|\n")
            for split, class_dist in distribution.items():
                total = sum(class_dist.values()) if class_dist else 0
                gara = class_dist.get('Gara', 0)
                naruto = class_dist.get('Naruto', 0)
                sakura = class_dist.get('Sakura', 0)
                tsunade = class_dist.get('Tsunade', 0)
                f.write(f"| {split.title()} | {total} | {gara} | {naruto} | {sakura} | {tsunade} |\n")
            f.write("\n## Model Configuration\n\n")
            model_info = eval_results['model_info']
            for key, value in model_info.items():
                if key != 'text_prompts_used':
                    f.write(f"- **{key.replace('_', ' ').title()}**: {value}\n")
            f.write("\n## Sample Predictions by Split\n\n")
            for split in ['train', 'valid', 'test']:
                split_samples = [s for s in per_sample_results if s.get('split') == split][:5]
                if split_samples:
                    f.write(f"### {split.title()} Split (First 5 samples)\n\n")
                    f.write("| Filename | True Class | Predicted | Confidence | Correct |\n")
                    f.write("|----------|------------|-----------|------------|----------|\n")
                    for sample in split_samples:
                        correct_emoji = "✅" if sample['correct'] else "❌"
                        f.write(f"| {sample['filename']} | {sample['true_class']} | "
                                f"{sample['predicted_class']} | {sample['confidence']:.3f} | {correct_emoji} |\n")
                    f.write("\n")
            f.write("## Error Analysis\n\n")
            incorrect_samples = [s for s in per_sample_results if not s['correct']]
            if incorrect_samples:
                f.write(f"**Total Errors**: {len(incorrect_samples)}\n\n")
                errors_by_split = {}
                for sample in incorrect_samples:
                    split = sample.get('split', 'unknown')
                    if split not in errors_by_split:
                        errors_by_split[split] = []
                    errors_by_split[split].append(sample)
                f.write("**Errors by Split**:\n")
                for split, errors in errors_by_split.items():
                    f.write(f"- {split.title()}: {len(errors)} errors\n")
                error_by_class = {}
                for sample in incorrect_samples:
                    true_class = sample['true_class']
                    pred_class = sample['predicted_class']
                    if true_class not in error_by_class:
                        error_by_class[true_class] = {}
                    if pred_class not in error_by_class[true_class]:
                        error_by_class[true_class][pred_class] = 0
                    error_by_class[true_class][pred_class] += 1
                f.write("\n**Common Misclassifications**:\n\n")
                for true_class, predictions in error_by_class.items():
                    f.write(f"- **{true_class}** confused with:\n")
                    for pred_class, count in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
                        f.write(f"  - {pred_class}: {count} times\n")
                    f.write("\n")
            else:
                f.write("Perfect zero-shot classification on full dataset! No errors found.\n")
            f.write("\n## Methodology\n\n")
            f.write("This evaluation uses zero-shot classification with pre-trained CLIP on the complete dataset:\n")
            f.write("- No fine-tuning on Naruto character data\n")
            f.write("- Full dataset coverage: train (297) + valid (64) + test (30)\n")
            f.write("- Uses multiple text prompt templates for robust classification\n")
            f.write("- Averages embeddings across templates for each class\n")
            f.write("- Provides comprehensive baseline performance before fine-tuning\n")
            f.write("- Split-wise analysis shows consistency across different data partitions\n")


if __name__ == "__main__":
    print("PRE-TRAINED CLIP BASELINE EVALUATION")
    print("=" * 70)
    print()
    data_dir = r"./Anime -Naruto-.v1i.multiclass"
    save_dir = "results_pretrained_baseline"
    model_name = "ViT-B/32"
    evaluator = PretrainedCLIPEvaluator(model_name=model_name)
    print("EVALUATION OPTIONS:")
    print("=" * 40)
    print("1. Test Set Only (30 samples) - Standard comparison")
    print("2. Full Dataset (391 samples) - Comprehensive baseline")
    print()
    evaluation_mode = "full"
    if evaluation_mode == "full":
        print("RUNNING FULL DATASET EVALUATION (Train + Valid + Test)")
        results = evaluator.evaluate_zero_shot_full_dataset(data_dir, save_dir)
        print("\n" + "="*70)
        print("FULL DATASET ZERO-SHOT EVALUATION COMPLETE")
        print("="*70)
        overall_metrics = results['overall_metrics']
        print(f"OVERALL ZERO-SHOT ACCURACY: {overall_metrics['zero_shot_accuracy']:.4f}")
        print(f"PERCENTAGE CORRECT: {overall_metrics['percentage_correct']:.2f}%")
        print(f"✅ CORRECT PREDICTIONS: {overall_metrics['correct_predictions']}")
        print(f"TOTAL SAMPLES: {overall_metrics['total_predictions']}")
        print(f"\nACCURACY BY SPLIT:")
        for split, metrics in results['split_accuracies'].items():
            accuracy_percent = metrics['accuracy'] * 100
            print(f"   {split.title():>5}: {accuracy_percent:>6.2f}% ({metrics['correct']}/{metrics['total']})")
        print(f"\nDETAILED RESULTS SAVED TO:")
        print(f"   Full Dataset Report: {save_dir}/reports/zero_shot_full_dataset_report.json")
        print(f"   Per-sample Predictions: {save_dir}/reports/zero_shot_full_dataset_predictions.json")
        print(f"   Confusion Matrix: {save_dir}/plots/zero_shot_full_dataset_confusion_matrix.png")
        print(f"   Summary Report: {save_dir}/reports/zero_shot_full_dataset_summary.md")
    else:
        print("RUNNING TEST SET EVALUATION (Test Only)")
        results = evaluator.evaluate_zero_shot(data_dir, save_dir)
        print("\n" + "="*70)
        print("TEST SET ZERO-SHOT EVALUATION COMPLETE")
        print("="*70)
        overall_metrics = results['overall_metrics']
        print(f"ZERO-SHOT ACCURACY: {overall_metrics['zero_shot_accuracy']:.4f}")
        print(f"PERCENTAGE CORRECT: {overall_metrics['percentage_correct']:.2f}%")
        print(f"✅ CORRECT PREDICTIONS: {overall_metrics['correct_predictions']}")
        print(f"TOTAL TEST SAMPLES: {overall_metrics['total_predictions']}")
        print(f"\nDETAILED RESULTS SAVED TO:")
        print(f"   Evaluation Report: {save_dir}/reports/zero_shot_evaluation_report.json")
        print(f"   Per-sample Predictions: {save_dir}/reports/zero_shot_per_sample_predictions.json")
        print(f"   Confusion Matrix: {save_dir}/plots/zero_shot_confusion_matrix.png")
        print(f"   Summary Report: {save_dir}/reports/zero_shot_evaluation_summary.md")
    print(f"\nPER-CLASS ZERO-SHOT ACCURACY:")
    for class_name, metrics in results['per_class_metrics'].items():
        accuracy_percent = metrics['accuracy'] * 100
        print(f"   {class_name:>8}: {accuracy_percent:>6.2f}% ({metrics['correct']}/{metrics['total']})")

    print(f"\nBASELINE EVALUATION COMPLETE!")
