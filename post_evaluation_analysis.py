import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from itertools import cycle


def calculate_and_print_metrics(y_true, y_pred, class_names, description="Results"):
    """Calculates and prints classification metrics."""
    accuracy = accuracy_score(y_true, y_pred)

    # Macro-average for overall P/R/F1, then per-class for detailed view
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)

    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{description} Metrics:")
    print(f"   Overall Accuracy: {accuracy:.3f}")
    print(f"   Macro Precision: {precision:.3f}, Macro Recall: {recall:.3f}, Macro F1: {f1:.3f}")

    for i, name in class_names.items():
        if i < len(precision_per_class):  # Ensure index is within bounds
            print(f"   {name} ({i}): P={precision_per_class[i]:.3f}, R={recall_per_class[i]:.3f}, F1={f1_per_class[i]:.3f}")

    print(f"   Confusion Matrix (True \u2193 | Predicted \u2192 ):")
    print(cm)
    print(f"   Classes: {[f'{k}={v}' for k,v in class_names.items()]}")

    return accuracy, precision, recall, f1, cm, precision_per_class, recall_per_class, f1_per_class


def plot_confusion_matrix(cm, class_names, title, save_path):
    """Plots a confusion matrix as a heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names.values(), yticklabels=class_names.values())
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_per_class_metrics(precision_per_class, recall_per_class, f1_per_class, class_names, title, save_path):
    """Plots per-class precision, recall, and F1-score as bar charts."""
    num_classes = len(class_names)
    if num_classes == 0:
        return  # Handle empty class_names

    metrics_df = pd.DataFrame({
        'Class': list(class_names.values()),
        'Precision': precision_per_class[:num_classes],
        'Recall': recall_per_class[:num_classes],
        'F1-Score': f1_per_class[:num_classes]
    })

    metrics_df_melted = metrics_df.melt(id_vars='Class', var_name='Metric', value_name='Score')

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Class', y='Score', hue='Metric', data=metrics_df_melted, palette='viridis')
    plt.title(title)
    plt.ylabel('Score')
    plt.xlabel('Class')
    plt.ylim(0, 1)  # Metrics are between 0 and 1
    plt.legend(title='Metric', loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_roc_curve(y_true, y_score, class_names, title, save_path):
    """Plots multi-class ROC curves."""
    n_classes = len(class_names)
    if n_classes == 0:
        return  # Handle empty class_names

    # Binarize the true labels
    y_true_bin = label_binarize(y_true, classes=list(class_names.keys()))
    if n_classes == 1:  # Handle binary case if passed as multi-class
        y_true_bin = np.array([1 if x == list(class_names.keys())[0] else 0 for x in y_true]).reshape(-1, 1)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure(figsize=(10, 8))
    colors = cycle(plt.cm.get_cmap('tab10').colors)
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {class_names[i]} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)  # Random classifier line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_precision_recall_curve(y_true, y_score, class_names, title, save_path):
    """Plots multi-class Precision-Recall curves."""
    n_classes = len(class_names)
    if n_classes == 0:
        return  # Handle empty class_names

    # Binarize the true labels
    y_true_bin = label_binarize(y_true, classes=list(class_names.keys()))
    if n_classes == 1:  # Handle binary case if passed as multi-class
        y_true_bin = np.array([1 if x == list(class_names.keys())[0] else 0 for x in y_true]).reshape(-1, 1)

    # Compute Precision-Recall curve and area for each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
        average_precision[i] = auc(recall[i], precision[i])

    # Plot all Precision-Recall curves
    plt.figure(figsize=(10, 8))
    colors = cycle(plt.cm.get_cmap('tab10').colors)
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label=f'Precision-Recall curve of class {class_names[i]} (AP = {average_precision[i]:.2f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def process_folder(folder_path, class_names):
    """
    Process all evaluation CSV files in a given folder.
    Plots are saved to an 'Analysis Plots' subdirectory inside folder_path.
    Returns a list of per-file metric dicts.
    """
    output_plots_dir = os.path.join(folder_path, "Analysis Plots")
    os.makedirs(output_plots_dir, exist_ok=True)

    folder_label = os.path.basename(folder_path)
    print(f"\n{'='*60}")
    print(f"Processing folder: {folder_label}")
    print(f"  Plots will be saved to: {output_plots_dir}")
    print(f"{'='*60}")

    # Find CSV files: supports both eval_*.csv and evaluation_results_*.csv naming
    csv_files = [f for f in os.listdir(folder_path)
                 if (f.startswith("eval_") or f.startswith("evaluation_results_")) and f.endswith(".csv")]

    if not csv_files:
        print(f"  No CSV files found in '{folder_path}'. Skipping.")
        return []

    print(f"  Found {len(csv_files)} CSV file(s).")

    all_file_metrics = []

    for csv_file in sorted(csv_files):
        file_path = os.path.join(folder_path, csv_file)
        print(f"\n  Processing: {csv_file}")

        try:
            df = pd.read_csv(file_path)
            y_true = df['true_label'].values
            y_pred_smoothed = df['smoothed_prediction'].values

            # Print metrics
            accuracy, precision, recall, f1, cm, precision_per_class, recall_per_class, f1_per_class = \
                calculate_and_print_metrics(
                    y_true, y_pred_smoothed, class_names,
                    description=f"{folder_label}/{csv_file} (Smoothed)"
                )

            # Store for overall summary
            all_file_metrics.append({
                'file': csv_file,
                'folder': folder_label,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })

            # Get probability columns
            prob_cols = [col for col in df.columns if col.endswith('_prob')]
            y_score = df[prob_cols].values

            base_name = os.path.splitext(csv_file)[0]

            # Plot Confusion Matrix
            plot_confusion_matrix(
                cm, class_names,
                title=f"Confusion Matrix\n{folder_label} | {csv_file} (Smoothed)",
                save_path=os.path.join(output_plots_dir, f"{base_name}_cm.png")
            )
            print(f"    [OK] Confusion Matrix saved.")

            # Plot Per-Class Metrics
            plot_per_class_metrics(
                precision_per_class, recall_per_class, f1_per_class, class_names,
                title=f"Per-Class Metrics\n{folder_label} | {csv_file} (Smoothed)",
                save_path=os.path.join(output_plots_dir, f"{base_name}_per_class_metrics.png")
            )
            print(f"    [OK] Per-Class Metrics saved.")

            # Plot ROC Curve
            plot_roc_curve(
                y_true, y_score, class_names,
                title=f"ROC Curve\n{folder_label} | {csv_file} (Smoothed)",
                save_path=os.path.join(output_plots_dir, f"{base_name}_roc_curve.png")
            )
            print(f"    [OK] ROC Curve saved.")

            # Plot Precision-Recall Curve
            plot_precision_recall_curve(
                y_true, y_score, class_names,
                title=f"Precision-Recall Curve\n{folder_label} | {csv_file} (Smoothed)",
                save_path=os.path.join(output_plots_dir, f"{base_name}_pr_curve.png")
            )
            print(f"    [OK] Precision-Recall Curve saved.")

        except Exception as e:
            print(f"    [ERROR] processing {csv_file}: {e}")
            import traceback
            traceback.print_exc()

    # ---- Per-folder summary across days ----
    if len(all_file_metrics) > 1:
        import re
        metrics_df = pd.DataFrame(all_file_metrics)

        def extract_day(filename):
            m = re.search(r'Day[_ ]?(\d+)', filename)
            return int(m.group(1)) if m else 0

        metrics_df['Day'] = metrics_df['file'].apply(extract_day)
        metrics_df = metrics_df.sort_values('Day')

        plt.figure(figsize=(10, 6))
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            plt.plot(metrics_df['Day'], metrics_df[metric], marker='o',
                     label=metric.replace('_', ' ').title())
        plt.title(f'Performance Across Evaluation Days — {folder_label}')
        plt.xlabel('Day')
        plt.ylabel('Score')
        plt.xticks(metrics_df['Day'])
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        summary_path = os.path.join(output_plots_dir, "overall_performance_across_days.png")
        plt.savefig(summary_path)
        plt.close()
        print(f"\n  [OK] Cross-day performance plot saved: {summary_path}")

    # Print folder summary
    print(f"\n  Summary for '{folder_label}':")
    for m in all_file_metrics:
        print(f"    {m['file']}: Acc={m['accuracy']:.3f}, P={m['precision']:.3f}, "
              f"R={m['recall']:.3f}, F1={m['f1']:.3f}")

    return all_file_metrics


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Post-Evaluation Analysis")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Path to a specific experiment results folder, "
                             "e.g. 'TamperingDetection/Evaluation Results/E1_SlowFast_R50_Gray'. "
                             "If omitted, ALL model sub-folders under the default "
                             "'Evaluation Results/' directory are processed.")
    parser.add_argument("--base_dir", type=str,
                        default="Evaluation Results",
                        help="Base directory that contains all model result sub-folders. "
                             "Only used when --results_dir is not supplied.")
    args = parser.parse_args()

    print("Post-Evaluation Analysis Script")
    print("================================")

    # 4-class for UHCTD
    class_names = {0: "Normal", 1: "Covered", 2: "Defocused", 3: "Moved"}

    # ----------- SINGLE FOLDER MODE -----------
    if args.results_dir:
        folder = args.results_dir
        if not os.path.isdir(folder):
            print(f"[ERROR] '{folder}' is not a valid directory.")
            return
        process_folder(folder, class_names)
        print("\nDone. Check the 'Analysis Plots' subfolder inside the specified directory.")
        return

    # ----------- MULTI-FOLDER (SCAN ALL) MODE -----------
    base_dir = args.base_dir
    if not os.path.isdir(base_dir):
        print(f"[ERROR] Base directory '{base_dir}' does not exist.")
        return

    print(f"Base directory  : {base_dir}")
    print("Scanning for model sub-folders and legacy top-level CSV files...\n")

    # Collect folders to process
    folders_to_process = []

    # 1. Check for legacy top-level CSV files
    legacy_csvs = [f for f in os.listdir(base_dir)
                   if (f.startswith("eval_") or f.startswith("evaluation_results_")) and f.endswith(".csv")]
    if legacy_csvs:
        folders_to_process.append(base_dir)  # process the top-level dir itself

    # 2. Find all model sub-folders that contain CSV files
    for entry in sorted(os.listdir(base_dir)):
        entry_path = os.path.join(base_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        # Skip the existing "Analysis Plots" folder and "Review Evaluation"
        if entry in ("Analysis Plots", "Review Evaluation"):
            continue
        # Check if this sub-folder has any CSV files
        sub_csvs = [f for f in os.listdir(entry_path)
                    if (f.startswith("eval_") or f.startswith("evaluation_results_")) and f.endswith(".csv")]
        if sub_csvs:
            folders_to_process.append(entry_path)

    if not folders_to_process:
        print(f"No evaluation CSV files found under '{base_dir}'. "
              f"Please ensure evaluation has been run.")
        return

    print(f"Found {len(folders_to_process)} folder(s) to process:")
    for fp in folders_to_process:
        print(f"  {fp}")

    # Process each folder
    all_model_metrics = {}
    for folder_path in folders_to_process:
        model_label = os.path.basename(folder_path)
        metrics = process_folder(folder_path, class_names)
        if metrics:
            all_model_metrics[model_label] = metrics

    # ---- Cross-model comparison summary ----
    print("\n\n" + "="*60)
    print("OVERALL SUMMARY ACROSS ALL MODELS")
    print("="*60)

    summary_rows = []
    for model_label, metrics_list in all_model_metrics.items():
        for m in metrics_list:
            summary_rows.append({
                'Model': model_label,
                'File': m['file'],
                'Accuracy': round(m['accuracy'], 3),
                'Precision': round(m['precision'], 3),
                'Recall': round(m['recall'], 3),
                'F1': round(m['f1'], 3),
            })
            print(f"[{model_label}] {m['file']}: "
                  f"Acc={m['accuracy']:.3f}, P={m['precision']:.3f}, "
                  f"R={m['recall']:.3f}, F1={m['f1']:.3f}")

    # Save summary CSV to base_dir
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_csv_path = os.path.join(base_dir, "all_models_summary.csv")
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"\nSummary CSV saved to: {summary_csv_path}")

        # Plot average F1 per model (mean across days)
        avg_df = summary_df.groupby('Model')[['Accuracy', 'Precision', 'Recall', 'F1']].mean().reset_index()
        avg_df = avg_df.sort_values('F1', ascending=False)

        plt.figure(figsize=(max(12, len(avg_df) * 0.8), 6))
        x = range(len(avg_df))
        width = 0.2
        for idx, metric in enumerate(['Accuracy', 'Precision', 'Recall', 'F1']):
            plt.bar([xi + idx * width for xi in x], avg_df[metric], width=width, label=metric)
        plt.xticks([xi + 1.5 * width for xi in x], avg_df['Model'], rotation=45, ha='right', fontsize=8)
        plt.ylabel('Score')
        plt.title('Average Performance per Model (Across Evaluation Days)')
        plt.legend()
        plt.ylim(0, 1)
        plt.tight_layout()
        comparison_path = os.path.join(base_dir, "model_comparison.png")
        plt.savefig(comparison_path)
        plt.close()
        print(f"Model comparison chart saved to: {comparison_path}")

    print("\nAnalysis complete. Check each model folder for 'Analysis Plots' sub-directories.")


if __name__ == "__main__":
    main()
