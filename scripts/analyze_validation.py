"""Quick script to analyze validation report."""
import pandas as pd

df = pd.read_csv('02_output/phase0_validation_report.csv')
ok = (df['Flag_Status'] == 'OK').sum()
total = len(df)
print(f'Parser Validation: {ok}/{total} OK ({ok/total*100:.1f}%)')
print(f'\nConfidence Distribution:')
print(df['Confidence'].value_counts().to_dict())
print(f'\nFlag Status Distribution:')
print(df['Flag_Status'].value_counts().to_dict())
print(f'\nTotal Headers Analyzed: {total}')
