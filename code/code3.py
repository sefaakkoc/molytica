from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model
model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Çıktı klasörü
output_subdir = os.path.join(images_dir, f'SVR_{timestamp}')
os.makedirs(output_subdir, exist_ok=True)

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
    f.write(f"SVR Sonuçları ({timestamp})\n")
    f.write(f"R²: {r2_score(y_test, y_pred):.3f}\nRMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}\n")