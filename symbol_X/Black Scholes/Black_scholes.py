import numpy as np
from scipy.stats import norm



def black_scholes(S, K, T, r, sigma, option_type):
    """
    S: Option price
    K: Strike price
    T: Time to expiration
    r: Risk-free rate
    sigma: Volatility (VIX)
    option_type: Call or Put
    """

    d1 = (np.log(S/K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return option_price




def main():

    s = float(input("Enter the option price: "))
    k = float(input("Enter the strike price: "))
    t = float(input("Enter the time to expiration in years: "))
    r = float(input("Enter the risk-free rate: "))
    sigma = float(input("Enter the volatility: "))
    o_type = input("Enter the option type (call/put): ")

    option_price = black_scholes(s, k, t, r, sigma, o_type)
    print(f"The price of the option is: {option_price}")


    return 0




if __name__ == "__main__":
    main()

