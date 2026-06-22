import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)

print("=" * 50)
print("  STEP 5: MODEL EVALUATION")
print("=" * 50)

with open("models/sentiment_model.pkl", "rb") as f:
    pipeline = pickle.load(f)

test_data = pd.read_csv("data/test_data.csv")
X_test = test_data['X_test']
y_test = test_data['y_test']

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\nAccuracy : {accuracy:.2%}")
print(f"F1 Score : {f1:.2%}")
print(f"\nDetailed Report:")
print(classification_report(y_test, y_pred))

labels = sorted(y_test.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels, ax=axes[0])
axes[0].set_title(f'Confusion Matrix\nAccuracy: {accuracy:.2%}', fontsize=13)
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')

report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df = report_df.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')

x = np.arange(len(report_df.index))
width = 0.25
axes[1].bar(x - width, report_df['precision'], width, label='Precision', color='steelblue')
axes[1].bar(x,         report_df['recall'],    width, label='Recall',    color='orange')
axes[1].bar(x + width, report_df['f1-score'],  width, label='F1 Score',  color='green')
axes[1].set_title('Precision / Recall / F1 per Class', fontsize=13)
axes[1].set_xticks(x)
axes[1].set_xticklabels(report_df.index)
axes[1].set_ylim(0, 1.1)
axes[1].legend()
axes[1].set_ylabel('Score')

plt.tight_layout()
plt.savefig("outputs/evaluation_report.png", dpi=150)
plt.show()
print("\nEvaluation plot saved: outputs/evaluation_report.png")

with open("outputs/evaluation_summary.txt", "w") as f:
    f.write("SENTIMENT PIPELINE - EVALUATION REPORT\n")
    f.write("=" * 40 + "\n\n")
    f.write(f"Accuracy  : {accuracy:.2%}\n")
    f.write(f"F1 Score  : {f1:.2%}\n\n")
    f.write("Detailed Classification Report:\n")
    f.write(classification_report(y_test, y_pred))

print("Summary saved: outputs/evaluation_summary.txt")