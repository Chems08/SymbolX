
# https://fr.tradingview.com/support/solutions/43000592270/#:~:text=Pour%20calculer%20l'EMA%2C%20il,par%201%20moins%20le%20multiplicateur.
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html

import yfinance as yf
import matplotlib.pyplot as plt

symbol = 'AAPL'
start='2024-05-19'
end='2024-08-23'

EMA_2 = []
smoothing = 3
days = 12

# Récupération des données
data = yf.download(symbol, start, end)

# Calcul de la moyenne mobile exponentielle sur une période de 12 jours
data['EMA'] = data['Close'].ewm(span=days, adjust=False).mean()



for i in range(len(data['Close'])):
    if i == 0:
        EMA_2.append(data['Close'].iloc[i])
    else:
        tmp = data['Close'].iloc[i]*(smoothing/(1+days)) + EMA_2[i-1]*(1-(smoothing/(1+days)))
        EMA_2.append(tmp)
data['EMA_2'] = EMA_2

# Affichage des données et ajoute le nom des abscisses et ordonnées

plt.figure(figsize=(12,6))
plt.xlabel('Date')
plt.ylabel('Prix')
plt.plot(data['Close'], label='Close Price', color = 'blue')
plt.plot(data['EMA'], label='EMA', color = 'red')
plt.plot(data['EMA_2'], label='EMA_2', color = 'green')
plt.title('Moyenne mobile exponentielle')
plt.legend()
plt.show()




