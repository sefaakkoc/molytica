import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime

# Klasör yapısı
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
images_dir = os.path.join(base_dir, 'static', 'images')
dataset_dir = os.path.join(base_dir, 'datasets')
outputs_dir = os.path.join(base_dir, 'static', 'outputs')

# Veri yükleme
data = pd.read_csv(os.path.join('out.csv'))
X = data[['molecular_fp_similarity', 'halogen_count', 'ewg_count', 'aromatic_rings']]
y = data['yield']

# Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Çıktı klasörü
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_subdir = os.path.join(images_dir, f'LinearRegression_{timestamp}')
os.makedirs(output_subdir, exist_ok=True)

# 1. Gerçek vs Tahmin
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.savefig(os.path.join(output_subdir, 'actual_vs_predicted.png'), dpi=300)
plt.close()

# 2. Artıklar
plt.figure(figsize=(10, 6))
sns.residplot(x=y_pred, y=y_test-y_pred, lowess=True)
plt.savefig(os.path.join(output_subdir, 'residuals.png'), dpi=300)
plt.close()

# 3. Q-Q Plot
plt.figure(figsize=(10, 6))
stats.probplot(y_test-y_pred, plot=plt)
plt.savefig(os.path.join(output_subdir, 'qq_plot.png'), dpi=300)
plt.close()

# 4. Aromatik Halka Etkisi
plt.figure(figsize=(10, 6))
sns.boxplot(x='aromatic_rings', y='yield', data=data)
plt.savefig(os.path.join(output_subdir, 'solvent_effect.png'), dpi=300)
plt.close()

# Rapor
with open(os.path.join(outputs_dir, 'asd.txt'), 'w') as f:
    f.write(f"Linear Regression Sonuçları ({timestamp})\n")
    f.write(f"R²: {r2_score(y_test, y_pred):.3f}\nRMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}\n")