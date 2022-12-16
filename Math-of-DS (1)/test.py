import pandas as pd

df = pd.read_csv('out1.csv')
df = df[df['Rental units affordable at  80% AMI (% of recently available units)'].isna()]

print(df.head(5))

