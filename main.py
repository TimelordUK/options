"""
Options Pricing & Greeks Explorer

An interactive tool for learning Black-Scholes option pricing and risk sensitivities.
Run with: streamlit run main.py
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import black_scholes as bs

st.set_page_config(
    page_title="Options Pricing Explorer",
    page_icon="📈",
    layout="wide",
)

st.title("Black-Scholes Options Pricing Explorer")

# Sidebar for input parameters
st.sidebar.header("Option Parameters")

S = st.sidebar.slider(
    "Spot Price (S)",
    min_value=50.0,
    max_value=150.0,
    value=100.0,
    step=1.0,
    help="Current price of the underlying asset",
)

K = st.sidebar.slider(
    "Strike Price (K)",
    min_value=50.0,
    max_value=150.0,
    value=100.0,
    step=1.0,
    help="Price at which the option can be exercised",
)

T = st.sidebar.slider(
    "Time to Expiry (years)",
    min_value=0.01,
    max_value=2.0,
    value=0.25,
    step=0.01,
    format="%.2f",
    help="Time until option expiration in years (0.25 = 3 months)",
)

r = st.sidebar.slider(
    "Risk-Free Rate (%)",
    min_value=0.0,
    max_value=15.0,
    value=5.0,
    step=0.25,
    help="Annualized risk-free interest rate",
) / 100  # Convert to decimal

sigma = st.sidebar.slider(
    "Volatility (%)",
    min_value=5.0,
    max_value=100.0,
    value=20.0,
    step=1.0,
    help="Annualized volatility (standard deviation of returns)",
) / 100  # Convert to decimal

option_type = st.sidebar.radio(
    "Option Type",
    ["Call", "Put"],
    help="Call = right to buy, Put = right to sell",
)

# Calculate current values
price = bs.option_price(S, K, T, r, sigma, option_type.lower())
greeks = bs.all_greeks(S, K, T, r, sigma, option_type.lower())

# Main content area
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Current Values")

    # Option price
    st.metric(
        label=f"{option_type} Option Price",
        value=f"${price:.4f}",
    )

    # Moneyness indicator
    moneyness = S / K
    if option_type == "Call":
        itm = S > K
    else:
        itm = S < K
    moneyness_label = "ITM" if itm else ("ATM" if abs(S - K) < 0.01 else "OTM")
    st.caption(f"Moneyness: {moneyness:.2%} ({moneyness_label})")

    st.divider()

    # Greeks display
    st.subheader("Greeks")

    greek_col1, greek_col2 = st.columns(2)

    with greek_col1:
        st.metric("Delta (Δ)", f"{greeks['delta']:.4f}")
        st.metric("Gamma (Γ)", f"{greeks['gamma']:.4f}")
        st.metric("Theta (Θ)", f"{greeks['theta']:.4f}")

    with greek_col2:
        st.metric("Vega (ν)", f"{greeks['vega']:.4f}")
        st.metric("Rho (ρ)", f"{greeks['rho']:.4f}")
        st.metric("Daily Theta", f"{greeks['theta']/365:.4f}")

with col2:
    st.subheader("Sensitivity Charts")

    # Create tabs for different chart types
    tab1, tab2, tab3, tab4 = st.tabs([
        "Price vs Spot",
        "Delta vs Spot",
        "Greeks vs Time",
        "Price vs Volatility"
    ])

    # Generate spot price range for charts
    spot_range = np.linspace(K * 0.5, K * 1.5, 100)

    with tab1:
        # Price vs Spot (with payoff at expiry for comparison)
        prices = [bs.option_price(s, K, T, r, sigma, option_type.lower()) for s in spot_range]
        intrinsic = [max(s - K, 0) if option_type == "Call" else max(K - s, 0) for s in spot_range]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=spot_range, y=prices,
            mode='lines', name='Option Price',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=spot_range, y=intrinsic,
            mode='lines', name='Intrinsic Value (Payoff at Expiry)',
            line=dict(color='red', width=2, dash='dash')
        ))
        fig.add_vline(x=K, line_dash="dot", line_color="gray", annotation_text="Strike")
        fig.add_vline(x=S, line_dash="solid", line_color="green", annotation_text="Current Spot")

        fig.update_layout(
            title=f"{option_type} Price vs Spot Price",
            xaxis_title="Spot Price",
            yaxis_title="Option Price",
            hovermode="x unified",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Understanding this chart"):
            st.markdown("""
            - **Blue line**: Current option price across different spot prices
            - **Red dashed line**: Intrinsic value (payoff if exercised immediately)
            - **Gap between lines**: Time value (the premium for optionality)
            - Time value is highest ATM and decays as we move ITM or OTM
            """)

    with tab2:
        # Delta vs Spot
        deltas = [bs.delta(s, K, T, r, sigma, option_type.lower()) for s in spot_range]
        gammas = [bs.gamma(s, K, T, r, sigma) for s in spot_range]

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(go.Scatter(
            x=spot_range, y=deltas,
            mode='lines', name='Delta',
            line=dict(color='blue', width=2)
        ), secondary_y=False)

        fig.add_trace(go.Scatter(
            x=spot_range, y=gammas,
            mode='lines', name='Gamma',
            line=dict(color='orange', width=2)
        ), secondary_y=True)

        fig.add_vline(x=K, line_dash="dot", line_color="gray", annotation_text="Strike")
        fig.add_vline(x=S, line_dash="solid", line_color="green", annotation_text="Current Spot")

        fig.update_layout(
            title="Delta and Gamma vs Spot Price",
            xaxis_title="Spot Price",
            hovermode="x unified",
            height=400,
        )
        fig.update_yaxes(title_text="Delta", secondary_y=False)
        fig.update_yaxes(title_text="Gamma", secondary_y=True)

        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Understanding this chart"):
            st.markdown("""
            - **Delta**: Rate of change of option price with spot. Call delta goes from 0 (deep OTM) to 1 (deep ITM).
            - **Gamma**: Rate of change of delta. Highest ATM where delta changes most rapidly.
            - High gamma near the strike = need frequent rehedging (gamma risk)
            - Delta ≈ probability of finishing ITM (approximately)
            """)

    with tab3:
        # Greeks vs Time to Expiry
        time_range = np.linspace(0.01, 2.0, 100)

        thetas = [bs.theta(S, K, t, r, sigma, option_type.lower()) for t in time_range]
        vegas = [bs.vega(S, K, t, r, sigma) for t in time_range]
        deltas_t = [bs.delta(S, K, t, r, sigma, option_type.lower()) for t in time_range]

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           subplot_titles=("Theta vs Time", "Vega vs Time"))

        fig.add_trace(go.Scatter(
            x=time_range, y=thetas,
            mode='lines', name='Theta (annual)',
            line=dict(color='red', width=2)
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=time_range, y=vegas,
            mode='lines', name='Vega',
            line=dict(color='purple', width=2)
        ), row=2, col=1)

        fig.add_vline(x=T, line_dash="solid", line_color="green", annotation_text="Current T")

        fig.update_layout(
            height=500,
            showlegend=True,
            hovermode="x unified",
        )
        fig.update_xaxes(title_text="Time to Expiry (years)", row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Understanding this chart"):
            st.markdown("""
            - **Theta**: Time decay - usually negative (options lose value over time)
            - Theta accelerates as expiry approaches - the famous "theta burn"
            - **Vega**: Sensitivity to volatility - decreases as expiry approaches
            - Long-dated options have more vega (vol has more time to matter)
            """)

    with tab4:
        # Price vs Volatility
        vol_range = np.linspace(0.05, 1.0, 100)

        prices_vol = [bs.option_price(S, K, T, r, v, option_type.lower()) for v in vol_range]
        vegas_vol = [bs.vega(S, K, T, r, v) for v in vol_range]

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(go.Scatter(
            x=vol_range * 100, y=prices_vol,
            mode='lines', name='Option Price',
            line=dict(color='blue', width=2)
        ), secondary_y=False)

        fig.add_trace(go.Scatter(
            x=vol_range * 100, y=vegas_vol,
            mode='lines', name='Vega',
            line=dict(color='purple', width=2, dash='dash')
        ), secondary_y=True)

        fig.add_vline(x=sigma * 100, line_dash="solid", line_color="green", annotation_text="Current Vol")

        fig.update_layout(
            title="Price and Vega vs Volatility",
            xaxis_title="Volatility (%)",
            hovermode="x unified",
            height=400,
        )
        fig.update_yaxes(title_text="Option Price", secondary_y=False)
        fig.update_yaxes(title_text="Vega", secondary_y=True)

        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Understanding this chart"):
            st.markdown("""
            - Option price increases with volatility (more uncertainty = more optionality value)
            - The relationship is roughly linear for ATM options
            - Vega (slope of price vs vol) varies with volatility level
            - This is why "vol of vol" matters in practice
            """)

# Scenario Explorer - interactive "what if" explanations
st.divider()
st.subheader("Scenario Explorer")
st.caption("Move the slider to explore different spot prices and understand why the option has that value.")

scenario_spot = st.slider(
    "What-if Spot Price",
    min_value=float(K * 0.5),
    max_value=float(K * 1.5),
    value=float(S),
    step=1.0,
    key="scenario_spot",
)

# Calculate values at scenario spot
scenario_price = bs.option_price(scenario_spot, K, T, r, sigma, option_type.lower())
scenario_greeks = bs.all_greeks(scenario_spot, K, T, r, sigma, option_type.lower())

if option_type == "Call":
    intrinsic_value = max(scenario_spot - K, 0)
    scenario_itm = scenario_spot > K
else:
    intrinsic_value = max(K - scenario_spot, 0)
    scenario_itm = scenario_spot < K

time_value = scenario_price - intrinsic_value
moneyness_pct = (scenario_spot / K - 1) * 100

# Determine moneyness category
if abs(scenario_spot - K) < K * 0.02:
    moneyness_cat = "ATM (At-The-Money)"
elif scenario_itm:
    if abs(moneyness_pct) > 20:
        moneyness_cat = "Deep ITM (In-The-Money)"
    else:
        moneyness_cat = "ITM (In-The-Money)"
else:
    if abs(moneyness_pct) > 20:
        moneyness_cat = "Deep OTM (Out-of-The-Money)"
    else:
        moneyness_cat = "OTM (Out-of-The-Money)"

# Helper to format small values with appropriate precision
def fmt_price(val):
    if val == 0:
        return "$0.00"
    elif abs(val) < 0.0001:
        return f"${val:.2e}"  # Scientific notation for very tiny
    elif abs(val) < 0.01:
        return f"${val:.6f}"  # 6 decimals for sub-penny
    else:
        return f"${val:.2f}"

def fmt_greek(val):
    if val == 0:
        return "0.00"
    elif abs(val) < 0.0001:
        return f"{val:.2e}"
    elif abs(val) < 0.01:
        return f"{val:.6f}"
    else:
        return f"{val:.4f}"

# Display breakdown
scen_col1, scen_col2 = st.columns(2)

with scen_col1:
    st.markdown(f"**Spot: ${scenario_spot:.0f} | Strike: ${K:.0f} | {option_type}**")
    st.markdown(f"**Status:** {moneyness_cat}")
    st.markdown(f"""
    | Component | Value |
    |-----------|-------|
    | Option Price | {fmt_price(scenario_price)} |
    | Intrinsic Value | {fmt_price(intrinsic_value)} |
    | Time Value | {fmt_price(time_value)} |
    | Delta | {fmt_greek(scenario_greeks['delta'])} |
    """)

with scen_col2:
    # Generate contextual explanation
    if option_type == "Call":
        if scenario_spot < K * 0.8:
            explanation = f"""
            **Why is this {option_type.lower()} nearly worthless?**

            With spot at ${scenario_spot:.0f} and strike at ${K:.0f}, you'd be paying ${K:.0f}
            for something worth ${scenario_spot:.0f} in the market. That's a ${K - scenario_spot:.0f} loss
            if exercised now.

            The small value (${scenario_price:.2f}) is pure **time value** - the chance that
            spot rises above ${K:.0f} before expiry. With {T*365:.0f} days left and {sigma*100:.0f}% volatility,
            there's a {abs(scenario_greeks['delta'])*100:.0f}% chance of finishing ITM.
            """
        elif scenario_spot < K:
            explanation = f"""
            **Why does this OTM {option_type.lower()} have value?**

            Spot (${scenario_spot:.0f}) is below strike (${K:.0f}), so **intrinsic value is zero** -
            no profit from immediate exercise.

            But there's ${time_value:.2f} of **time value**. With {T*365:.0f} days and {sigma*100:.0f}% vol,
            the market prices in a {abs(scenario_greeks['delta'])*100:.0f}% probability of spot rising
            above ${K:.0f}. The closer to the strike, the higher this probability.
            """
        elif abs(scenario_spot - K) < K * 0.02:
            explanation = f"""
            **ATM - Maximum Uncertainty**

            Spot (${scenario_spot:.0f}) equals strike (${K:.0f}). The option could go either way.

            **Time value is maximized** at ${time_value:.2f} because uncertainty is highest here.
            Delta ≈ {scenario_greeks['delta']:.0f}% means roughly 50/50 chance of finishing ITM.

            This is also where **gamma is highest** ({scenario_greeks['gamma']:.3f}) - small spot moves
            cause big delta changes. Hedging is most difficult here.
            """
        elif scenario_spot < K * 1.2:
            explanation = f"""
            **ITM - Has Real Value Now**

            Spot (${scenario_spot:.0f}) > Strike (${K:.0f}), so exercising gives ${intrinsic_value:.2f} profit.

            The option is worth ${scenario_price:.2f}:
            - **Intrinsic**: ${intrinsic_value:.2f} (locked-in value)
            - **Time value**: ${time_value:.2f} (could go higher still)

            Delta is {scenario_greeks['delta']:.2f} - the option moves almost 1:1 with spot but
            with downside protection (can't lose more than the premium).
            """
        else:
            explanation = f"""
            **Deep ITM - Behaves Like Stock**

            With spot at ${scenario_spot:.0f} vs strike ${K:.0f}, this is ${scenario_spot - K:.0f} in-the-money.

            **Delta ≈ {scenario_greeks['delta']:.2f}** means it moves nearly 1:1 with the underlying.
            Almost all value is **intrinsic** (${intrinsic_value:.2f}), minimal time value (${time_value:.2f}).

            At this point, you're essentially holding leveraged stock exposure with
            limited downside. The option will almost certainly be exercised.
            """
    else:  # Put
        if scenario_spot > K * 1.2:
            explanation = f"""
            **Why is this {option_type.lower()} nearly worthless?**

            With spot at ${scenario_spot:.0f} and strike at ${K:.0f}, you'd be selling at ${K:.0f}
            something worth ${scenario_spot:.0f}. That's a ${scenario_spot - K:.0f} loss vs market price.

            The small value (${scenario_price:.2f}) is pure **time value** - the chance that
            spot falls below ${K:.0f} before expiry. With {T*365:.0f} days left,
            there's only a {abs(scenario_greeks['delta'])*100:.0f}% chance of finishing ITM.
            """
        elif scenario_spot > K:
            explanation = f"""
            **Why does this OTM {option_type.lower()} have value?**

            Spot (${scenario_spot:.0f}) is above strike (${K:.0f}), so **intrinsic value is zero** -
            no profit from immediate exercise.

            But there's ${time_value:.2f} of **time value**. With {T*365:.0f} days and {sigma*100:.0f}% vol,
            the market prices in a {abs(scenario_greeks['delta'])*100:.0f}% probability of spot falling
            below ${K:.0f}. Puts provide insurance against downside.
            """
        elif abs(scenario_spot - K) < K * 0.02:
            explanation = f"""
            **ATM - Maximum Uncertainty**

            Spot (${scenario_spot:.0f}) equals strike (${K:.0f}). The option could go either way.

            **Time value is maximized** at ${time_value:.2f} because uncertainty is highest here.
            Delta ≈ {scenario_greeks['delta']:.2f} means roughly 50/50 chance of finishing ITM.

            This is also where **gamma is highest** ({scenario_greeks['gamma']:.3f}) - small spot moves
            cause big delta changes.
            """
        elif scenario_spot > K * 0.8:
            explanation = f"""
            **ITM - Has Real Value Now**

            Spot (${scenario_spot:.0f}) < Strike (${K:.0f}), so exercising gives ${intrinsic_value:.2f} profit.

            The option is worth ${scenario_price:.2f}:
            - **Intrinsic**: ${intrinsic_value:.2f} (locked-in value)
            - **Time value**: ${time_value:.2f} (could go higher still)

            Delta is {scenario_greeks['delta']:.2f} - the option gains as spot falls,
            providing portfolio protection.
            """
        else:
            explanation = f"""
            **Deep ITM - High Insurance Value**

            With spot at ${scenario_spot:.0f} vs strike ${K:.0f}, this is ${K - scenario_spot:.0f} in-the-money.

            **Delta ≈ {scenario_greeks['delta']:.2f}** means it moves nearly 1:1 (inverse) with spot.
            Almost all value is **intrinsic** (${intrinsic_value:.2f}), minimal time value (${time_value:.2f}).

            This put provides strong downside protection - it's like insurance that's
            already paying out.
            """

    st.markdown(explanation)

# Educational section
st.divider()

with st.expander("Black-Scholes Formula Explained", expanded=False):
    st.markdown(r"""
    ### The Black-Scholes Formula

    For a **European call option**:

    $$C = S \cdot N(d_1) - K \cdot e^{-rT} \cdot N(d_2)$$

    For a **European put option**:

    $$P = K \cdot e^{-rT} \cdot N(-d_2) - S \cdot N(-d_1)$$

    Where:

    $$d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}$$

    $$d_2 = d_1 - \sigma\sqrt{T}$$

    **Intuition:**
    - $N(d_2)$ ≈ probability of expiring in-the-money (risk-neutral)
    - $N(d_1)$ is $N(d_2)$ adjusted for the delta hedge
    - $S \cdot N(d_1)$: Expected value of receiving the stock
    - $K \cdot e^{-rT} \cdot N(d_2)$: PV of expected cost if exercised

    **Key Assumptions:**
    - Log-normal stock prices (geometric Brownian motion)
    - Constant volatility and interest rates
    - No dividends, no transaction costs
    - Continuous trading possible
    """)

with st.expander("Greeks Quick Reference", expanded=False):
    st.markdown("""
    | Greek | Symbol | Measures | Call Sign | Put Sign |
    |-------|--------|----------|-----------|----------|
    | **Delta** | Δ | Price sensitivity to spot | + (0 to 1) | - (-1 to 0) |
    | **Gamma** | Γ | Delta sensitivity to spot | + | + |
    | **Theta** | Θ | Time decay (per year) | - (usually) | - (usually) |
    | **Vega** | ν | Sensitivity to volatility | + | + |
    | **Rho** | ρ | Sensitivity to interest rate | + | - |

    **Interview Tips:**
    - Delta ≈ probability of expiring ITM (approximately)
    - Gamma is highest for ATM options near expiry
    - Theta accelerates as expiry approaches ("gamma scalping" exploits this)
    - Vega is highest for ATM, long-dated options
    - Put-call parity: C - P = S - K·e^(-rT)
    """)

# Footer
st.divider()
st.caption("Built for learning Black-Scholes and options Greeks. Adjust parameters in the sidebar and watch the charts update.")
