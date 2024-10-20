import pandas as pd

def clean_data(df):
    # #Etape 1 - v0.1
    # #Drop rows with missing values
    # df = df.dropna()
    
    # #Drop duplicate rows
    df = df.drop_duplicates()

    # #Etape 2 - v0.3
    #supprime les 0 de la collonne Market Cap (in M) et Volume 52 weeks et Volume 1 month
    df = df[df['Market Cap (in M)'] != 0]
    df = df[df['Volume 52 weeks'] != 0]
    df = df[df['Volume 1 month'] != 0]

    #supprime les lignes avec une valeur manquante dans la colonne Symbol et beta
    df = df.dropna(subset=['Symbol', 'Beta', 'Sector', 'Industry', 'Currency'])

    #supprime la colonne Chiffre d'affaire, Dividend Per Share Annual, EBITDA CAGR (5y), EBITDA, Dividend Yield Indicated Annual, P/E Ratio
    df = df.drop(columns=["Chiffre d'affaires", 'Dividend Per Share Annual', 'EBITDA CAGR (5y)', 'EBITDA', 'Dividend Yield Indicated Annual', 'P/E Ratio'])

    # #Etape 3 - v0.4
    # supprime les valeurs manquantes dans Performance (52 weeks)=185, Price 52 Weeks Ago=161, Résultat net=4, EPS Annual=5, ROI Annual=34
    df = df.dropna(subset=['Performance (52 weeks)', 'Price 52 Weeks Ago', 'Résultat net', 'EPS Annual', 'ROI Annual'])

    #supprime la colonne 'Ratio Debt/Equity (Annual)'
    df = df.drop(columns=['Ratio Debt/Equity (Annual)'])

    #supprime les lignes avec des 0 dans EPS Annual, ROI Annual
    df = df[df['EPS Annual'] != 0]
    df = df[df['ROI Annual'] != 0]

    #Etape 4 - v0.5
    #supprime les valeurs manquantes dans Total assets
    df = df.dropna(subset=['Total assets'])




    return df

# Load data
data = pd.read_csv('/pre-processing/cleaned_data_v0.4.csv')

# Clean data
cleaned_data = clean_data(data)

# Save cleaned data
cleaned_data.to_csv('/pre-processing/cleaned_data_v0.5.csv', index=False)