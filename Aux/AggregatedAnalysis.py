import pandas as pd
import joblib
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Passo 1: Carregar o Dataset
df = pd.read_csv('datasets/CICIDS-2017/dataset.csv')

# Supondo que as colunas de características sejam 'feature1', 'feature2', ..., e a coluna 'Label' seja o alvo
X = df.drop(columns=['Label']).values  # Características
y = df['Label'].values  # Etiquetas

# Passo 2: Carregar os Modelos Salvos
# Carregar o modelo supervisionado (Random Forest)
supervised_model_path = 'saves/supervised/rf_model.pkl'
clf = joblib.load(supervised_model_path)

# Carregar o modelo "Self" (exemplo com KMeans)
self_model_path = 'saves/self/final_model.plk'  # Supondo que o modelo "Self" também tenha sido salvo
self_model = joblib.load(self_model_path)

# Passo 3: Avaliar o Modelo Supervisionado
y_pred_supervised = clf.predict(X)

# Relatório de classificação para o modelo supervisionado
report_supervised = classification_report(y, y_pred_supervised, output_dict=True)
print("Classification Report for Supervised Model:")
print(report_supervised)

# Passo 4: Avaliar o Modelo "Self"
y_pred_self = self_model.predict(X)

# Relatório de classificação para o modelo "Self"
report_self = classification_report(y, y_pred_self, output_dict=True)
print("Classification Report for Self Model:")
print(report_self)

# Passo 5: Comparação entre os Modelos
# Extraindo as métricas de precisão de cada relatório
accuracy_supervised = report_supervised['accuracy']
accuracy_self = report_self['accuracy']

precision_supervised = report_supervised['macro avg']['precision']
precision_self = report_self['macro avg']['precision']

recall_supervised = report_supervised['macro avg']['recall']
recall_self = report_self['macro avg']['recall']

f1_score_supervised = report_supervised['macro avg']['f1-score']
f1_score_self = report_self['macro avg']['f1-score']

# Exibindo as comparações de precisão, recall e f1-score
metrics = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Supervised': [accuracy_supervised, precision_supervised, recall_supervised, f1_score_supervised],
    'Self': [accuracy_self, precision_self, recall_self, f1_score_self]
}

comparison_df = pd.DataFrame(metrics)
print("\nComparison between Supervised and Self Models:")
print(comparison_df)

# Passo 6: Plotando as Comparações
fig, ax = plt.subplots(figsize=(8, 6))
comparison_df.set_index('Metric').plot(kind='bar', ax=ax, color=['blue', 'orange'])

ax.set_title('Comparison of Supervised and Self Models')
ax.set_ylabel('Score')
ax.set_xticklabels(comparison_df['Metric'], rotation=0)
plt.tight_layout()
plt.show()
