import pandas as pd
from ydata_profiling import ProfileReport


df = pd.read_csv(f'Bureau/pre-processing/brouillon/cleaned_asset_no_na_and_duplicates.csv')

profile = ProfileReport(df, title='Pandas Profiling Report', explorative=True)

profile.to_file(f'Bureau/pre-processing/assets_report_no_na_and_duplicates.html')