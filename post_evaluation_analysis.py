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
        if i < len(precision_per_class): # Ensure index is within bounds
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
    if num_classes == 0: return # Handle empty class_names

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
    plt.ylim(0, 1) # Metrics are between 0 and 1
    plt.legend(title='Metric', loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(y_true, y_score, class_names, title, save_path):
    """Plots multi-class ROC curves."""
    n_classes = len(class_names)
    if n_classes == 0: return # Handle empty class_names

    # Binarize the true labels
    y_true_bin = label_binarize(y_true, classes=list(class_names.keys()))
    if n_classes == 1: # Handle binary case if passed as multi-class
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

    plt.plot([0, 1], [0, 1], 'k--', lw=2) # Random classifier line
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
    if n_classes == 0: return # Handle empty class_names

    # Binarize the true labels
    y_true_bin = label_binarize(y_true, classes=list(class_names.keys()))
    if n_classes == 1: # Handle binary case if passed as multi-class
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

def main():
    print("Post-Evaluation Analysis Script")
    print("================================")

    results_dir = "./Evaluation Results"
    output_plots_dir = os.path.join(results_dir, "Analysis Plots")
    os.makedirs(output_plots_dir, exist_ok=True)

    # Assuming 4-class for current context
    class_names = {0: "Normal", 1: "Covered", 2: "Defocused", 3: "Moved"}

    all_day_metrics = [] # To store metrics for overall summary

    # Dynamically find all evaluation result CSVs
    csv_files = [f for f in os.listdir(results_dir) if f.startswith("evaluation_results_") and f.endswith(".csv")]
    
    if not csv_files:
        print(f"No CSV files found in {results_dir}. Please ensure evaluation has been run.")
        return

    print(f"Found {len(csv_files)} evaluation result files in '{results_dir}'.")

    for csv_file in sorted(csv_files): # Process in sorted order (e.g., Day 3, Day 4...)
        file_path = os.path.join(results_dir, csv_file)
        print(f"\nProcessing file: {csv_file}")
        
        try:
            df = pd.read_csv(file_path)
            y_true = df['true_label'].values
            y_pred_smoothed = df['smoothed_prediction'].values

            # Print metrics
            accuracy, precision, recall, f1, cm, precision_per_class, recall_per_class, f1_per_class = calculate_and_print_metrics(
                y_true, y_pred_smoothed, class_names, description=f"Metrics for {csv_file} (Smoothed)"
            )
            
            # Store for overall summary
            all_day_metrics.append({
                'file': csv_file,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })

            # Get probability columns
            prob_cols = [col for col in df.columns if col.endswith('_prob')]
            y_score = df[prob_cols].values

            # Plot Confusion Matrix
            plot_confusion_matrix(
                cm, class_names, 
                title=f"Confusion Matrix for {csv_file} (Smoothed)",
                save_path=os.path.join(output_plots_dir, f"{os.path.splitext(csv_file)[0]}_cm.png")
            )
            print(f"  Confusion Matrix plot saved to {output_plots_dir}")

            # Plot Per-Class Metrics
            plot_per_class_metrics(
                precision_per_class, recall_per_class, f1_per_class, class_names,
                title=f"Per-Class Metrics for {csv_file} (Smoothed)",
                save_path=os.path.join(output_plots_dir, f"{os.path.splitext(csv_file)[0]}_per_class_metrics.png")
            )
            print(f"  Per-Class Metrics plot saved to {output_plots_dir}")

            # Plot ROC Curve
            plot_roc_curve(
                y_true, y_score, class_names,
                title=f"ROC Curve for {csv_file} (Smoothed)",
                save_path=os.path.join(output_plots_dir, f"{os.path.splitext(csv_file)[0]}_roc_curve.png")
            )
            print(f"  ROC Curve plot saved to {output_plots_dir}")

            # Plot Precision-Recall Curve
            plot_precision_recall_curve(
                y_true, y_score, class_names,
                title=f"Precision-Recall Curve for {csv_file} (Smoothed)",
                save_path=os.path.join(output_plots_dir, f"{os.path.splitext(csv_file)[0]}_pr_curve.png")
            )
            print(f"  Precision-Recall Curve plot saved to {output_plots_dir}")

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            import traceback
            traceback.print_exc()

    # Overall Summary
    print("\n\n================================")
    print("OVERALL METRICS SUMMARY ACROSS DAYS")
    print("================================")
    for metrics in all_day_metrics:
        print(f"File: {metrics['file']}")
        print(f"  Accuracy: {metrics['accuracy']:.3f}, P: {metrics['precision']:.3f}, R: {metrics['recall']:.3f}, F1: {metrics['f1']:.3f}")
    
    # Optional: Plot overall trends if needed
    if len(all_day_metrics) > 1:
        metrics_df = pd.DataFrame(all_day_metrics)
        metrics_df['Day'] = metrics_df['file'].apply(lambda x: int(x.split('_Day ')[1].split('.')[0]))
        metrics_df = metrics_df.sort_values('Day')

        plt.figure(figsize=(10, 6))
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            plt.plot(metrics_df['Day'], metrics_df[metric], marker='o', label=metric.replace('_', ' ').title())
        plt.title('Performance Metrics Across Evaluation Days')
        plt.xlabel('Day')
        plt.ylabel('Score')
        plt.xticks(metrics_df['Day'])
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        summary_plot_path = os.path.join(output_plots_dir, "overall_performance_across_days.png")
        plt.savefig(summary_plot_path)
        plt.close()
        print(f"Overall performance plot saved to {summary_plot_path}")

    print("\nAnalysis complete. Check 'Analysis Plots' folder for visualizations.")

if __name__ == "__main__":
    main()
