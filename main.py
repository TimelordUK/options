"""
Options Pricing & Greeks Explorer

An interactive tool for learning Black-Scholes option pricing and risk sensitivities.
Run with: streamlit run main.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import lognorm, norm
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
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Price vs Spot",
        "Delta vs Spot",
        "Greeks vs Time",
        "Price vs Volatility",
        "Animated Vol",
        "Animated Delta",
        "Monte Carlo"
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

    with tab5:
        # Animated chart showing how price curve changes with volatility
        st.caption("Watch how the option price curve shape changes as volatility increases. Press Play to animate.")

        # Create volatility range for animation frames
        vol_steps = np.linspace(0.05, 0.80, 30)  # 5% to 80% vol in 30 frames

        # Build animation frames
        frames = []
        for v in vol_steps:
            prices_at_vol = [bs.option_price(s, K, T, r, v, option_type.lower()) for s in spot_range]
            frames.append(go.Frame(
                data=[go.Scatter(
                    x=spot_range,
                    y=prices_at_vol,
                    mode='lines',
                    line=dict(color='blue', width=2),
                    name='Option Price'
                )],
                name=f"{v*100:.0f}%",
                layout=go.Layout(title=f"{option_type} Price vs Spot (Vol = {v*100:.0f}%)")
            ))

        # Initial frame data
        initial_prices = [bs.option_price(s, K, T, r, vol_steps[0], option_type.lower()) for s in spot_range]
        intrinsic = [max(s - K, 0) if option_type == "Call" else max(K - s, 0) for s in spot_range]

        fig_anim = go.Figure(
            data=[
                go.Scatter(
                    x=spot_range,
                    y=initial_prices,
                    mode='lines',
                    line=dict(color='blue', width=2),
                    name='Option Price'
                ),
                go.Scatter(
                    x=spot_range,
                    y=intrinsic,
                    mode='lines',
                    line=dict(color='red', width=2, dash='dash'),
                    name='Intrinsic Value'
                ),
            ],
            frames=frames
        )

        # Add play/pause buttons and slider
        fig_anim.update_layout(
            title=f"{option_type} Price vs Spot (Vol = {vol_steps[0]*100:.0f}%)",
            xaxis_title="Spot Price",
            yaxis_title="Option Price",
            yaxis=dict(range=[0, max(intrinsic) * 1.3]),  # Fixed y-axis for smooth animation
            height=450,
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    y=1.15,
                    x=0.0,
                    xanchor="left",
                    buttons=[
                        dict(
                            label="▶ Play",
                            method="animate",
                            args=[None, {
                                "frame": {"duration": 150, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 50}
                            }]
                        ),
                        dict(
                            label="⏸ Pause",
                            method="animate",
                            args=[[None], {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0}
                            }]
                        )
                    ]
                )
            ],
            sliders=[{
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 14},
                    "prefix": "Volatility: ",
                    "visible": True,
                    "xanchor": "center"
                },
                "transition": {"duration": 50},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.05,
                "y": 0,
                "steps": [
                    {"args": [[f"{v*100:.0f}%"],
                              {"frame": {"duration": 0, "redraw": True},
                               "mode": "immediate",
                               "transition": {"duration": 0}}],
                     "label": f"{v*100:.0f}%",
                     "method": "animate"}
                    for v in vol_steps
                ]
            }]
        )

        fig_anim.add_vline(x=K, line_dash="dot", line_color="gray", annotation_text="Strike")

        st.plotly_chart(fig_anim, use_container_width=True)

        with st.expander("Understanding this animation"):
            st.markdown("""
            **What you're seeing:**
            - As volatility increases, the option price curve "lifts" across all spot prices
            - Deep OTM options gain value fastest (percentage-wise) because higher vol = more chance of reaching strike
            - The red dashed intrinsic value line stays fixed - it's the payoff at expiry regardless of vol
            - The gap between blue and red is **time value**, which expands with volatility

            **Key insight:** This is why options are sometimes called "volatility instruments" -
            buying an option is essentially a bet that realized vol will exceed implied vol.
            """)

    with tab6:
        # Animated delta chart showing how delta curve changes with volatility
        st.markdown("""
        **Delta = N(d₁)** where d₁ = [ln(S/K) + (r + σ²/2)T] / **(σ√T)**

        The denominator **σ√T** is the key: it controls how sharply delta transitions from 0 to 1.
        """)

        # Create volatility range for animation
        delta_vol_steps = np.linspace(0.05, 0.80, 30)  # 5% to 80% vol

        # Build animation frames for delta
        delta_frames = []
        for v in delta_vol_steps:
            deltas_at_vol = [bs.delta(s, K, T, r, v, option_type.lower()) for s in spot_range]
            gammas_at_vol = [bs.gamma(s, K, T, r, v) for s in spot_range]
            delta_frames.append(go.Frame(
                data=[
                    go.Scatter(
                        x=spot_range,
                        y=deltas_at_vol,
                        mode='lines',
                        line=dict(color='blue', width=2),
                        name='Delta'
                    ),
                    go.Scatter(
                        x=spot_range,
                        y=gammas_at_vol,
                        mode='lines',
                        line=dict(color='orange', width=2),
                        name='Gamma',
                        yaxis='y2'
                    )
                ],
                name=f"{v*100:.0f}%",
                layout=go.Layout(title=f"Delta & Gamma vs Spot (Vol = {v*100:.0f}%)")
            ))

        # Initial frame
        initial_deltas = [bs.delta(s, K, T, r, delta_vol_steps[0], option_type.lower()) for s in spot_range]
        initial_gammas = [bs.gamma(s, K, T, r, delta_vol_steps[0]) for s in spot_range]

        fig_delta_anim = go.Figure(
            data=[
                go.Scatter(
                    x=spot_range,
                    y=initial_deltas,
                    mode='lines',
                    line=dict(color='blue', width=2),
                    name='Delta'
                ),
                go.Scatter(
                    x=spot_range,
                    y=initial_gammas,
                    mode='lines',
                    line=dict(color='orange', width=2),
                    name='Gamma',
                    yaxis='y2'
                )
            ],
            frames=delta_frames
        )

        # Fixed y-axis ranges for smooth animation
        y1_range = [-1.1, 1.1] if option_type == "Put" else [-0.1, 1.1]

        fig_delta_anim.update_layout(
            title=f"Delta & Gamma vs Spot (Vol = {delta_vol_steps[0]*100:.0f}%)",
            xaxis_title="Spot Price",
            yaxis=dict(title="Delta", range=y1_range),
            yaxis2=dict(
                title="Gamma",
                overlaying='y',
                side='right',
                range=[0, max(initial_gammas) * 2.5]
            ),
            height=450,
            legend=dict(x=0.01, y=0.99),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    y=1.15,
                    x=0.0,
                    xanchor="left",
                    buttons=[
                        dict(
                            label="▶ Play",
                            method="animate",
                            args=[None, {
                                "frame": {"duration": 150, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 50}
                            }]
                        ),
                        dict(
                            label="⏸ Pause",
                            method="animate",
                            args=[[None], {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0}
                            }]
                        )
                    ]
                )
            ],
            sliders=[{
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 14},
                    "prefix": "Volatility: ",
                    "visible": True,
                    "xanchor": "center"
                },
                "transition": {"duration": 50},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.05,
                "y": 0,
                "steps": [
                    {"args": [[f"{v*100:.0f}%"],
                              {"frame": {"duration": 0, "redraw": True},
                               "mode": "immediate",
                               "transition": {"duration": 0}}],
                     "label": f"{v*100:.0f}%",
                     "method": "animate"}
                    for v in delta_vol_steps
                ]
            }]
        )

        fig_delta_anim.add_vline(x=K, line_dash="dot", line_color="gray", annotation_text="Strike")

        st.plotly_chart(fig_delta_anim, use_container_width=True)

        with st.expander("Understanding Delta vs Volatility", expanded=True):
            st.markdown(r"""
            **Why does the delta curve flatten with higher volatility?**

            The formula: **d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)**

            | Volatility | σ√T (denominator) | Effect on d₁ | Delta curve shape |
            |------------|-------------------|--------------|-------------------|
            | **5%** | 0.025 (tiny) | d₁ swings wildly | Sharp step function |
            | **20%** | 0.10 | d₁ moderate | S-curve |
            | **80%** | 0.40 (large) | d₁ changes slowly | Nearly flat line |

            **The intuition:**
            - **Low vol**: Market is confident about where spot ends up. Near the strike,
              a tiny move tips from "probably OTM" to "probably ITM" → delta jumps sharply.
            - **High vol**: Anything can happen. Deep OTM? Still has a chance. Deep ITM?
              Might not stay there. So delta is uncertain (close to 0.5) across a wide range.

            **Watch the gamma (orange):**
            - Gamma = slope of the delta curve = how fast delta changes
            - Low vol → gamma spikes sharply at strike (hard to hedge!)
            - High vol → gamma is low and spread out (easier to hedge)

            **Why this matters for trading:**
            - Delta hedging at low vol near strike requires constant rebalancing (high gamma)
            - This is "gamma risk" - you're always chasing the hedge
            - Market makers charge more (wider spreads) for high-gamma positions
            """)

        with st.expander("d₁ Formula Breakdown - Every Term Explained"):
            st.markdown(r"""
            ### d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)

            | Term | Name | Meaning |
            |------|------|---------|
            | **S** | Spot price | Current price of the underlying (e.g., $100) |
            | **K** | Strike price | Price at which you can exercise (e.g., $100) |
            | **ln(S/K)** | Log-moneyness | How far ITM/OTM you are in log terms |
            | **r** | Risk-free rate | Annualized interest rate (e.g., 0.05 = 5%) |
            | **σ** | Volatility | Annualized std dev of returns (e.g., 0.20 = 20%) |
            | **T** | Time to expiry | In years (0.25 = 3 months) |
            | **σ√T** | Vol × √time | Total expected price spread by expiry |
            | **σ²/2** | Convexity adjustment | Corrects for log-normal vs normal returns |

            ---

            **The numerator: ln(S/K) + (r + σ²/2)T**

            This is asking: *"Where do we expect the stock to be at expiry?"*

            - `ln(S/K)` = where you are now (0 if ATM, positive if ITM, negative if OTM)
            - `(r + σ²/2)T` = expected drift (stock grows at risk-free rate + volatility adjustment)

            **The denominator: σ√T**

            This normalizes everything into *"number of standard deviations"*

            - Higher vol → larger denominator → d₁ closer to 0 → delta closer to 0.5
            - More time → larger denominator → same effect
            - This is why long-dated ATM options have delta ≈ 0.5-0.6 regardless of vol

            ---

            **Example calculation (ATM, 3-month, 20% vol, 5% rate):**

            ```
            S = 100, K = 100, T = 0.25, r = 0.05, σ = 0.20

            ln(S/K) = ln(1) = 0
            (r + σ²/2)T = (0.05 + 0.02)×0.25 = 0.0175
            σ√T = 0.20 × 0.5 = 0.10

            d₁ = (0 + 0.0175) / 0.10 = 0.175

            Delta = N(0.175) = 0.569
            ```

            So an ATM call has delta ≈ 0.57, slightly above 0.5 due to the drift term.
            """)

    with tab7:
        # Monte Carlo simulation showing probability distribution approach
        st.markdown("""
        **Option Pricing as Expected Value**

        Instead of formulas, think of it this way:
        1. Simulate many possible future prices
        2. Calculate payoff for each: max(S - K, 0) for calls
        3. Average the payoffs
        4. Discount to today

        This is exactly what Black-Scholes computes analytically!
        """)

        # Number of simulations slider
        n_sims = st.select_slider(
            "Number of simulations",
            options=[100, 500, 1000, 5000, 10000, 50000, 100000],
            value=10000,
            help="More simulations = closer to theoretical price (but slower)"
        )

        # Run simulation button to avoid constant recalculation
        if st.button("Run Simulation", type="primary"):
            # Run Monte Carlo simulation
            np.random.seed(None)  # Random seed each time

            # Stock follows: S_T = S * exp((r - σ²/2)T + σ√T × Z)
            Z = np.random.normal(0, 1, n_sims)
            S_T = S * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)

            # Calculate payoffs
            if option_type == "Call":
                payoffs = np.maximum(S_T - K, 0)
            else:
                payoffs = np.maximum(K - S_T, 0)

            # Monte Carlo price
            mc_price = np.exp(-r*T) * np.mean(payoffs)
            bs_price_exact = bs.option_price(S, K, T, r, sigma, option_type.lower())

            # Store results in session state
            st.session_state['mc_results'] = {
                'Z': Z,
                'S_T': S_T,
                'payoffs': payoffs,
                'mc_price': mc_price,
                'bs_price': bs_price_exact,
                'n_sims': n_sims,
                'K': K,
                'S': S,
                'T': T,
                'r': r,
                'option_type': option_type
            }

        # Display results if available
        if 'mc_results' in st.session_state:
            res = st.session_state['mc_results']
            S_T = res['S_T']
            payoffs = res['payoffs']
            mc_price = res['mc_price']
            bs_price_exact = res['bs_price']

            # Results metrics
            mc_col1, mc_col2, mc_col3 = st.columns(3)
            with mc_col1:
                st.metric("Monte Carlo Price", f"${mc_price:.4f}")
            with mc_col2:
                st.metric("Black-Scholes Exact", f"${bs_price_exact:.4f}")
            with mc_col3:
                error = mc_price - bs_price_exact
                st.metric("Difference", f"${error:+.4f}",
                         delta=f"{error/bs_price_exact*100:+.2f}%" if bs_price_exact > 0 else "N/A")

            # Show detailed calculation for small number of simulations
            if res['n_sims'] <= 100:
                with st.expander(f"Step-by-Step Calculation ({res['n_sims']} paths)", expanded=True):
                    st.markdown(f"""
                    **Formula:** S_T = S × exp[(r - σ²/2)T + σ√T × Z]  where Z ~ N(0,1)

                    With S={res['S']}, r={res['r']:.2%}, σ={sigma:.2%}, T={res['T']:.2f}:
                    """)

                    # Build the calculation table
                    Z_vals = res['Z']
                    calc_data = []
                    for i in range(len(Z_vals)):
                        calc_data.append({
                            'Path': i + 1,
                            'Random Z': f"{Z_vals[i]:+.4f}",
                            'Final Price (S_T)': f"${S_T[i]:.2f}",
                            'Payoff': f"${payoffs[i]:.2f}",
                            'Status': 'ITM' if payoffs[i] > 0 else 'OTM'
                        })

                    df = pd.DataFrame(calc_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)

                    # Summary calculation
                    st.markdown("---")
                    st.markdown(f"""
                    **Calculation:**

                    | Step | Calculation | Result |
                    |------|-------------|--------|
                    | 1. Sum of all payoffs | Σ payoffs | ${np.sum(payoffs):.2f} |
                    | 2. Divide by {res['n_sims']} paths | ${np.sum(payoffs):.2f} ÷ {res['n_sims']} | ${np.mean(payoffs):.4f} |
                    | 3. Discount factor | e^(-{res['r']:.2%} × {res['T']:.2f}) | {np.exp(-res['r']*res['T']):.6f} |
                    | 4. **Option Price** | ${np.mean(payoffs):.4f} × {np.exp(-res['r']*res['T']):.4f} | **${mc_price:.4f}** |

                    Paths ITM: {np.sum(payoffs > 0)} / {res['n_sims']} = {np.mean(payoffs > 0)*100:.1f}%
                    """)

            # Create the visualization
            fig_mc = make_subplots(
                rows=2, cols=1,
                subplot_titles=(
                    f"Distribution of Simulated Prices at Expiry ({res['n_sims']:,} paths)",
                    f"{res['option_type']} Payoff: Probability × Payoff = Option Value"
                ),
                vertical_spacing=0.15,
                row_heights=[0.5, 0.5]
            )

            # Histogram of final prices
            fig_mc.add_trace(
                go.Histogram(
                    x=S_T,
                    nbinsx=80,
                    name='Simulated Prices',
                    marker_color='steelblue',
                    opacity=0.7
                ),
                row=1, col=1
            )

            # Add theoretical log-normal distribution
            price_range = np.linspace(S * 0.3, S * 2.0, 200)
            # Log-normal parameters
            mu_ln = np.log(S) + (r - 0.5*sigma**2)*T
            sigma_ln = sigma * np.sqrt(T)
            theoretical_pdf = lognorm.pdf(price_range, s=sigma_ln, scale=np.exp(mu_ln))
            # Scale to match histogram
            hist_scale = len(S_T) * (S_T.max() - S_T.min()) / 80
            fig_mc.add_trace(
                go.Scatter(
                    x=price_range,
                    y=theoretical_pdf * hist_scale,
                    mode='lines',
                    name='Theoretical (Log-Normal)',
                    line=dict(color='red', width=2)
                ),
                row=1, col=1
            )

            # Add strike line
            fig_mc.add_vline(x=res['K'], line_dash="dash", line_color="green",
                           annotation_text=f"Strike={res['K']}", row=1, col=1)

            # Payoff diagram with probability shading
            # Sort for clean visualization
            sort_idx = np.argsort(S_T)
            S_T_sorted = S_T[sort_idx]
            payoffs_sorted = payoffs[sort_idx]

            # Bin the data for cleaner visualization
            n_bins = 100
            bin_edges = np.linspace(S_T.min(), S_T.max(), n_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_counts, _ = np.histogram(S_T, bins=bin_edges)
            bin_probs = bin_counts / len(S_T)

            # Calculate average payoff in each bin
            bin_payoffs = np.zeros(n_bins)
            for i in range(n_bins):
                mask = (S_T >= bin_edges[i]) & (S_T < bin_edges[i+1])
                if mask.sum() > 0:
                    bin_payoffs[i] = payoffs[mask].mean()

            # Expected payoff contribution per bin
            expected_contrib = bin_probs * bin_payoffs

            # Plot payoff function
            if res['option_type'] == "Call":
                payoff_line = np.maximum(bin_centers - res['K'], 0)
            else:
                payoff_line = np.maximum(res['K'] - bin_centers, 0)

            fig_mc.add_trace(
                go.Scatter(
                    x=bin_centers,
                    y=payoff_line,
                    mode='lines',
                    name='Payoff Function',
                    line=dict(color='blue', width=2)
                ),
                row=2, col=1
            )

            # Shade the expected value contribution
            fig_mc.add_trace(
                go.Bar(
                    x=bin_centers,
                    y=expected_contrib * 100,  # Scale for visibility
                    name='Prob × Payoff (scaled)',
                    marker_color='rgba(0, 150, 0, 0.5)',
                    width=(bin_edges[1] - bin_edges[0]) * 0.9
                ),
                row=2, col=1
            )

            fig_mc.add_vline(x=res['K'], line_dash="dash", line_color="green", row=2, col=1)

            fig_mc.update_layout(
                height=600,
                showlegend=True,
                legend=dict(x=0.7, y=0.98)
            )
            fig_mc.update_xaxes(title_text="Price at Expiry", row=1, col=1)
            fig_mc.update_xaxes(title_text="Price at Expiry", row=2, col=1)
            fig_mc.update_yaxes(title_text="Frequency", row=1, col=1)
            fig_mc.update_yaxes(title_text="Payoff / Expected Contribution", row=2, col=1)

            st.plotly_chart(fig_mc, use_container_width=True)

            # Statistics
            with st.expander("Simulation Statistics", expanded=True):
                prob_itm = np.mean(payoffs > 0) * 100
                d2_val = bs.d2(S, K, T, r, sigma)
                theoretical_prob_itm = norm.cdf(d2_val) * 100 if option_type == "Call" else norm.cdf(-d2_val) * 100

                st.markdown(f"""
                | Statistic | Simulated | Theoretical |
                |-----------|-----------|-------------|
                | **Probability of finishing ITM** | {prob_itm:.1f}% | {theoretical_prob_itm:.1f}% (N(d₂)) |
                | **Average payoff (if ITM)** | ${np.mean(payoffs[payoffs > 0]):.2f} | - |
                | **Expected payoff (undiscounted)** | ${np.mean(payoffs):.4f} | - |
                | **Option price (discounted)** | ${mc_price:.4f} | ${bs_price_exact:.4f} |

                **Convergence:** With {res['n_sims']:,} simulations, error is ${abs(mc_price - bs_price_exact):.4f}
                ({abs(mc_price - bs_price_exact)/bs_price_exact*100:.2f}%).
                Try increasing simulations to see the price converge to Black-Scholes!
                """)

        else:
            st.info("Click 'Run Simulation' to generate Monte Carlo paths and visualize the probability distribution.")

        with st.expander("How Monte Carlo Pricing Works"):
            st.markdown(r"""
            **The Algorithm:**

            1. **Model the stock price** using Geometric Brownian Motion:

               $S_T = S_0 \times e^{(r - \frac{\sigma^2}{2})T + \sigma\sqrt{T} \times Z}$

               where Z is a random draw from N(0,1)

            2. **Generate N random paths** (e.g., 100,000)

            3. **Calculate payoff for each path:**
               - Call: max(S_T - K, 0)
               - Put: max(K - S_T, 0)

            4. **Average the payoffs** = Expected payoff at expiry

            5. **Discount to today:** Price = e^(-rT) × Average payoff

            **Why it works:**
            - Law of Large Numbers: sample average → true expected value
            - Black-Scholes IS this expectation, solved analytically
            - Monte Carlo is numerical, BS is closed-form - same answer!

            **Why use Monte Carlo?**
            - Works for complex options where no formula exists
            - Path-dependent options (Asian, barriers, lookbacks)
            - Multiple underlying assets
            - Any payoff function you can code
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
