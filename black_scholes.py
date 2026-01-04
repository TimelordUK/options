"""
Black-Scholes Option Pricing Model

This module implements the Black-Scholes formula for European options pricing
and the associated Greeks (sensitivities).

The Black-Scholes formula assumes:
- European options (exercise only at expiry)
- No dividends
- Constant volatility and risk-free rate
- Log-normal distribution of stock prices
- No transaction costs or taxes
- Continuous trading

Key formula components:
- d1 and d2 are standardized measures of moneyness adjusted for drift and volatility
- N(x) is the cumulative normal distribution function
- The call price is: S*N(d1) - K*e^(-rT)*N(d2)
- The put price uses put-call parity or: K*e^(-rT)*N(-d2) - S*N(-d1)
"""

import numpy as np
from scipy.stats import norm


def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate d1 parameter in Black-Scholes formula.

    d1 = [ln(S/K) + (r + sigma^2/2)*T] / (sigma * sqrt(T))

    Interpretation: d1 is related to the probability of exercise in a
    risk-neutral world, adjusted for the delta hedge ratio.

    Parameters:
        S: Current stock/spot price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free interest rate (annualized, e.g., 0.05 for 5%)
        sigma: Volatility (annualized, e.g., 0.20 for 20%)
    """
    if T <= 0:
        return np.inf if S > K else (-np.inf if S < K else 0)

    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate d2 parameter in Black-Scholes formula.

    d2 = d1 - sigma * sqrt(T)

    Interpretation: N(d2) is the risk-neutral probability that the option
    expires in-the-money (i.e., S > K for a call at expiry).
    """
    if T <= 0:
        return np.inf if S > K else (-np.inf if S < K else 0)

    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate Black-Scholes call option price.

    C = S * N(d1) - K * e^(-rT) * N(d2)

    Where:
    - S * N(d1): Expected value of receiving stock if option exercised
    - K * e^(-rT) * N(d2): Present value of paying strike if exercised

    The formula gives the fair price assuming no arbitrage opportunities.
    """
    if T <= 0:
        return max(S - K, 0)  # Intrinsic value at expiry

    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)

    return S * norm.cdf(d1_val) - K * np.exp(-r * T) * norm.cdf(d2_val)


def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate Black-Scholes put option price.

    P = K * e^(-rT) * N(-d2) - S * N(-d1)

    Alternatively, can be derived from put-call parity:
    P = C - S + K * e^(-rT)
    """
    if T <= 0:
        return max(K - S, 0)  # Intrinsic value at expiry

    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)

    return K * np.exp(-r * T) * norm.cdf(-d2_val) - S * norm.cdf(-d1_val)


def delta(S: float, K: float, T: float, r: float, sigma: float,
          option_type: str = "call") -> float:
    """
    Calculate option Delta.

    Delta = dV/dS (rate of change of option price with respect to spot)

    Call Delta = N(d1)  [ranges from 0 to 1]
    Put Delta = N(d1) - 1 = -N(-d1)  [ranges from -1 to 0]

    Interpretation:
    - Delta ≈ probability of expiring ITM (approximately, not exactly)
    - Delta tells you how many shares to hold to hedge the option
    - ATM options have delta ≈ 0.5 (call) or -0.5 (put)
    """
    if T <= 0:
        if option_type.lower() == "call":
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0

    d1_val = d1(S, K, T, r, sigma)

    if option_type.lower() == "call":
        return norm.cdf(d1_val)
    else:
        return norm.cdf(d1_val) - 1


def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate option Gamma.

    Gamma = d(Delta)/dS = d²V/dS² (rate of change of delta with respect to spot)

    Gamma = N'(d1) / (S * sigma * sqrt(T))

    Where N'(x) is the standard normal PDF.

    Interpretation:
    - Gamma is the same for calls and puts (put-call parity)
    - Highest for ATM options near expiry
    - Measures convexity/curvature of the option payoff
    - High gamma = delta changes rapidly = harder to hedge
    """
    if T <= 0:
        return 0.0

    d1_val = d1(S, K, T, r, sigma)
    return norm.pdf(d1_val) / (S * sigma * np.sqrt(T))


def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate option Vega.

    Vega = dV/d(sigma) (sensitivity to volatility)

    Vega = S * N'(d1) * sqrt(T)

    Note: Often quoted per 1% change in vol (divide by 100).

    Interpretation:
    - Vega is the same for calls and puts
    - Highest for ATM options with longer time to expiry
    - Long options are long vega (benefit from vol increase)
    - Vega decreases as expiry approaches
    """
    if T <= 0:
        return 0.0

    d1_val = d1(S, K, T, r, sigma)
    return S * norm.pdf(d1_val) * np.sqrt(T)


def theta(S: float, K: float, T: float, r: float, sigma: float,
          option_type: str = "call") -> float:
    """
    Calculate option Theta (per year).

    Theta = dV/dT (sensitivity to time - usually negative, "time decay")

    For a call:
    Theta = -[S * N'(d1) * sigma / (2 * sqrt(T))] - r * K * e^(-rT) * N(d2)

    For a put:
    Theta = -[S * N'(d1) * sigma / (2 * sqrt(T))] + r * K * e^(-rT) * N(-d2)

    Interpretation:
    - Usually negative (options lose value over time)
    - Most negative for ATM options near expiry
    - Theta accelerates as expiry approaches ("theta burn")
    - Deep ITM puts can have positive theta (due to interest rate effect)

    Note: Divide by 365 to get daily theta.
    """
    if T <= 0:
        return 0.0

    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)

    # First term is the same for calls and puts
    term1 = -(S * norm.pdf(d1_val) * sigma) / (2 * np.sqrt(T))

    if option_type.lower() == "call":
        term2 = -r * K * np.exp(-r * T) * norm.cdf(d2_val)
    else:
        term2 = r * K * np.exp(-r * T) * norm.cdf(-d2_val)

    return term1 + term2


def rho(S: float, K: float, T: float, r: float, sigma: float,
        option_type: str = "call") -> float:
    """
    Calculate option Rho.

    Rho = dV/dr (sensitivity to interest rate)

    Call Rho = K * T * e^(-rT) * N(d2)
    Put Rho = -K * T * e^(-rT) * N(-d2)

    Interpretation:
    - Calls have positive rho (higher rates = higher call value)
    - Puts have negative rho (higher rates = lower put value)
    - More significant for longer-dated options
    - Often quoted per 1% change in rate (divide by 100)
    """
    if T <= 0:
        return 0.0

    d2_val = d2(S, K, T, r, sigma)

    if option_type.lower() == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2_val)
    else:
        return -K * T * np.exp(-r * T) * norm.cdf(-d2_val)


def option_price(S: float, K: float, T: float, r: float, sigma: float,
                 option_type: str = "call") -> float:
    """Convenience function to get call or put price."""
    if option_type.lower() == "call":
        return call_price(S, K, T, r, sigma)
    else:
        return put_price(S, K, T, r, sigma)


def all_greeks(S: float, K: float, T: float, r: float, sigma: float,
               option_type: str = "call") -> dict:
    """Calculate all Greeks at once and return as a dictionary."""
    return {
        "delta": delta(S, K, T, r, sigma, option_type),
        "gamma": gamma(S, K, T, r, sigma),
        "theta": theta(S, K, T, r, sigma, option_type),
        "vega": vega(S, K, T, r, sigma),
        "rho": rho(S, K, T, r, sigma, option_type),
    }
