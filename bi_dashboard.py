import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

rng = np.random.default_rng(35)
n = 240
df = pd.DataFrame({
    'country': rng.choice(['India','USA','Germany','Japan'], n),
    'category': rng.choice(['Electronics','Clothing','Grocery'], n),
    'sales': rng.normal(1200, 350, n).clip(100).round(2),
    'date': pd.date_range('2026-01-01', periods=n, freq='D')
})
df.loc[[25, 95, 175], 'sales'] *= 5
model = IsolationForest(contamination=0.04, random_state=35)
df['anomaly'] = model.fit_predict(df[['sales']]) == -1
print('SALES BY COUNTRY')
print(df.groupby('country')['sales'].agg(['sum','mean']).round(2))
print('\nAnomalies detected:', df['anomaly'].sum())
print('\nMANAGEMENT INSIGHTS')
top = df.groupby('country')['sales'].sum().idxmax()
print(f'Highest revenue country: {top}')
print(f'Total revenue: {df.sales.sum():,.2f}')
print(f'Investigate {df.anomaly.sum()} unusually sized transactions.')
fig, ax = plt.subplots(1, 2, figsize=(12,4))
df.groupby('country')['sales'].sum().plot.bar(ax=ax[0], title='Revenue by Country')
ax[1].plot(df['date'], df['sales'], lw=0.8)
ax[1].scatter(df.loc[df.anomaly,'date'],df.loc[df.anomaly,'sales'],c='red',label='Anomaly')
ax[1].set_title('Sales & AI Anomaly Detection')
ax[1].legend()
plt.tight_layout()
plt.show()
