import math
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ------------------------------------------------------------
# Binomial Option Lab — polished Streamlit browser app
# ------------------------------------------------------------
st.set_page_config(
    page_title="Binomial Option Lab",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

BG = "#0b1020"
PANEL = "#151d30"
BORDER = "#273247"
TEXT = "#e5eaf5"
MUTED = "#9ca9bf"
BLUE = "#69a7ff"
ORANGE = "#f4a277"
TEAL = "#44d3b0"
PURPLE = "#b69aff"
PINK = "#ff7896"

st.markdown(f"""
<style>
.stApp {{ background: {BG}; color: {TEXT}; }}
[data-testid="stSidebar"] {{ background: #101727; border-right: 1px solid {BORDER}; }}
.block-container {{ max-width: 1500px; padding-top: 1.35rem; padding-bottom: 3rem; }}
h1, h2, h3, h4 {{ color: {TEXT}; letter-spacing: -0.02em; }}
small, .muted {{ color: {MUTED}; }}
.kpi-card {{
    background: linear-gradient(135deg, #171f33 0%, #131a2b 100%);
    border: 1px solid {BORDER}; border-left: 4px solid var(--accent);
    border-radius: 13px; padding: 15px 17px 16px 17px;
    min-height: 104px; box-shadow: 0 5px 18px rgba(0,0,0,.10);
}}
.kpi-label {{ color: {MUTED}; font-size: 12px; line-height: 1.4; margin-bottom: 9px; }}
.kpi-value {{ color: var(--accent); font-size: clamp(20px, 2vw, 29px); font-weight: 750; line-height: 1.15; font-variant-numeric: tabular-nums; }}
.kpi-sub {{ color: #7f8da7; font-size: 11px; padding-top: 8px; }}
.section-intro {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 12px; padding: 13px 16px; margin: 6px 0 10px 0; }}
.section-intro p {{ margin: 4px 0 0 0; color: {MUTED}; font-size: 13px; }}
div[data-testid="stMetric"] {{ background: {PANEL}; border: 1px solid {BORDER}; padding: 12px 15px; border-radius: 12px; }}
div[data-testid="stMetricLabel"] {{ color: {MUTED}; }}
div[data-testid="stMetricValue"] {{ color: {TEXT}; }}
.math-card {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 12px; padding: 16px 18px; margin: 9px 0; }}
.math-card h4 {{ margin: 0 0 7px 0; color: {BLUE}; font-size: 15px; }}
.math-card p {{ color: {MUTED}; font-size: 13px; margin: 4px 0 10px 0; }}
div[data-testid="stAlert"] {{ border-radius: 10px; }}
hr {{ border-color: {BORDER}; }}
</style>
""", unsafe_allow_html=True)


# -------------------- PRICING FUNCTIONS --------------------
def payoff(stock, strike, kind):
    stock = np.asarray(stock, dtype=float)
    if kind == "Call":
        return np.maximum(stock - strike, 0.0)
    return np.maximum(strike - stock, 0.0)


def crr_params(r, sigma, T, N):
    dt = T / N
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(r * dt) - d) / (u - d)
    if not 0.0 <= p <= 1.0:
        raise ValueError(
            "Risk-neutral probability is outside [0, 1]. Try more lattice steps, "
            "higher volatility, or different inputs."
        )
    return dt, u, d, p, math.exp(-r * dt)


def price_tree(S, K, r, sigma, T, N, kind, style):
    dt, u, d, p, disc = crr_params(r, sigma, T, N)
    stocks, values, intrinsic, continuation, exercise = {}, {}, {}, {}, {}

    j = np.arange(N + 1)
    terminal_stock = S * (u ** j) * (d ** (N - j))
    terminal_payoff = payoff(terminal_stock, K, kind)
    stocks[N] = terminal_stock
    values[N] = terminal_payoff.copy()
    intrinsic[N] = terminal_payoff.copy()
    continuation[N] = np.full(N + 1, np.nan)
    exercise[N] = np.zeros(N + 1, dtype=bool)

    for i in range(N - 1, -1, -1):
        j = np.arange(i + 1)
        stock = S * (u ** j) * (d ** (i - j))
        intr = payoff(stock, K, kind)
        cont = disc * (
            p * values[i + 1][1:] + (1.0 - p) * values[i + 1][:-1]
        )
        ex = ((intr > cont + 1e-10) & (intr > 0)) if style == "American" else np.zeros(i + 1, dtype=bool)
        val = np.maximum(intr, cont) if style == "American" else cont
        stocks[i], intrinsic[i], continuation[i], exercise[i], values[i] = stock, intr, cont, ex, val

    return {
        "price": float(values[0][0]), "dt": dt, "u": u, "d": d, "p": p, "disc": disc,
        "stocks": stocks, "values": values, "intrinsic": intrinsic,
        "continuation": continuation, "exercise": exercise,
    }


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes(S, K, r, sigma, T, kind):
    if T <= 0:
        return float(payoff([S], K, kind)[0])
    if sigma <= 0:
        return math.exp(-r * T) * float(payoff([S * math.exp(r * T)], K, kind)[0])
    root_t = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * root_t)
    d2 = d1 - sigma * root_t
    if kind == "Call":
        return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)


def greeks_finite_difference(S, K, r, sigma, T, N, kind, style):
    h = max(S * 0.001, 0.01)
    base = price_tree(S, K, r, sigma, T, N, kind, style)["price"]
    up = price_tree(S + h, K, r, sigma, T, N, kind, style)["price"]
    down = price_tree(max(S - h, 1e-8), K, r, sigma, T, N, kind, style)["price"]
    delta = (up - down) / (2.0 * h)
    gamma = (up - 2.0 * base + down) / (h * h)
    vol_step = 0.01
    vol_up = price_tree(S, K, r, sigma + vol_step, T, N, kind, style)["price"]
    vol_down = price_tree(S, K, r, max(sigma - vol_step, 1e-6), T, N, kind, style)["price"]
    vega_per_1pct = (vol_up - vol_down) / 2.0
    day = min(1.0 / 365.0, T / 4.0)
    theta_day = float("nan")
    if T > day:
        less_time = price_tree(S, K, r, sigma, T - day, N, kind, style)["price"]
        theta_day = less_time - base
    return delta, gamma, vega_per_1pct, theta_day


def european_price_fast(S, K, r, sigma, T, N, kind):
    # Risk-neutral terminal expectation, vectorised over terminal nodes.
    dt, u, d, p, _ = crr_params(r, sigma, T, N)
    j = np.arange(N + 1)
    terminal_stock = S * (u ** j) * (d ** (N - j))
    probs = np.empty(N + 1, dtype=float)
    if p <= 0.0:
        probs[:] = 0.0
        probs[0] = 1.0
    elif p >= 1.0:
        probs[:] = 0.0
        probs[-1] = 1.0
    else:
        probs[0] = (1.0 - p) ** N
        for k in range(N):
            probs[k + 1] = probs[k] * (N - k) / (k + 1) * p / (1.0 - p)
    return math.exp(-r * T) * float(np.dot(probs, payoff(terminal_stock, K, kind)))


# -------------------- CHARTS --------------------
def lattice_figure(tree, N, kind):
    edge_x, edge_y = [], []
    node_x, node_y, node_values, node_custom = [], [], [], []
    ex_x, ex_y = [], []
    for i in range(N):
        for j in range(i + 1):
            y = 2 * j - i
            edge_x += [i, i + 1, None, i, i + 1, None]
            edge_y += [y, y - 1, None, y, y + 1, None]

    for i in range(N + 1):
        for j in range(i + 1):
            y = 2 * j - i
            s = float(tree["stocks"][i][j])
            v = float(tree["values"][i][j])
            intr = float(tree["intrinsic"][i][j])
            cont = float(tree["continuation"][i][j]) if i < N else float("nan")
            node_x.append(i)
            node_y.append(y)
            node_values.append(v)
            node_custom.append([i, j, s, v, intr, cont])
            if tree["exercise"][i][j]:
                ex_x.append(i)
                ex_y.append(y)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(color="rgba(125,140,170,0.25)", width=1.1),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers", name="Nodes",
        showlegend=False,
        marker=dict(
            size=max(7, 13 - N * 0.12), color=node_values,
            colorscale=[[0.0, "#440154"], [0.25, "#3b528b"], [0.5, "#21918c"], [0.75, "#5ec962"], [1.0, "#fde725"]],
            showscale=True,
            colorbar=dict(
                title=dict(text="Option value", side="top", font=dict(color="#dbe4f5", size=12)),
                tickfont=dict(color="#c5cede", size=10), thickness=15,
                len=0.82, x=1.02, outlinewidth=0,
            ),
            line=dict(width=0.5, color="rgba(255,255,255,0.16)"),
        ),
        customdata=node_custom,
        hovertemplate=(
            "<b>Step %{customdata[0]}</b><br>Up moves: %{customdata[1]}"
            "<br>Stock price: %{customdata[2]:.4f}"
            "<br><b>Option value: %{customdata[3]:.4f}</b>"
            "<br>Exercise value: %{customdata[4]:.4f}"
            "<br>Continuation value: %{customdata[5]:.4f}<extra></extra>"
        ),
    ))
    if ex_x:
        fig.add_trace(go.Scatter(
            x=ex_x, y=ex_y, mode="markers", name="Early exercise",
            showlegend=False,
            marker=dict(symbol="circle-open", size=21 if N <= 15 else 14, color=PINK,
                        line=dict(width=2.2, color=PINK)),
            hovertemplate="Early-exercise node<extra></extra>",
        ))

    fig.update_layout(
        template="plotly_dark", height=590,
        margin=dict(l=25, r=95, t=55, b=45),
        paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        title=dict(text=f"Recombining CRR lattice · {kind} · N = {N}", font=dict(size=17, color=TEXT), x=0.02),
        xaxis_title="Time step i", yaxis_title="Net up-moves (2j − i)",
        xaxis=dict(dtick=max(1, math.ceil(N / 15)), gridcolor="#29344a", zeroline=False),
        yaxis=dict(gridcolor="#29344a", zeroline=False),
        showlegend=False,
    )
    return fig


def convergence_figure(S, K, r, sigma, T, max_n, kind):
    ns, vals = [], []
    for n in range(1, max_n + 1):
        try:
            vals.append(european_price_fast(S, K, r, sigma, T, n, kind))
            ns.append(n)
        except (ValueError, OverflowError, FloatingPointError):
            continue
    bs = black_scholes(S, K, r, sigma, T, kind)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ns, y=vals, mode="lines", name="European binomial",
        line=dict(color=BLUE, width=2),
        hovertemplate="Steps: %{x}<br>Binomial price: %{y:.6f}<extra></extra>",
    ))
    fig.add_hline(
        y=bs, line=dict(color=ORANGE, dash="dash", width=2),
        annotation_text=f"Black–Scholes = {bs:.4f}",
        annotation_font_color=ORANGE,
        annotation_position="top right",
    )
    fig.update_layout(
        template="plotly_dark", height=370,
        margin=dict(l=25, r=25, t=55, b=45),
        paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        title=dict(text=f"European binomial → Black–Scholes convergence · {kind}", font=dict(size=17, color=TEXT), x=0.02),
        xaxis_title="Number of time steps N", yaxis_title="Option price",
        xaxis=dict(gridcolor="#29344a", zeroline=False),
        yaxis=dict(gridcolor="#29344a", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=TEXT)),
    )
    return fig, bs, ns, vals


def kpi_card(label, value, sub, accent):
    return f"""
    <div class="kpi-card" style="--accent:{accent};">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-sub">{sub}</div>
    </div>
    """


# -------------------- INPUTS --------------------
st.title("◈ Binomial Option Lab")
st.caption("OPTION PRICING WORKBENCH  ·  CRR LATTICE  ·  GREEKS  ·  BLACK–SCHOLES  ·  RISK-NEUTRAL VALUATION")

with st.sidebar:
    st.markdown("## INPUTS")
    st.caption("Adjust an input to recalculate the model.")
    S0 = st.number_input("Spot S₀", min_value=0.01, value=100.0, step=1.0, format="%.2f")
    K = st.number_input("Strike K", min_value=0.01, value=100.0, step=1.0, format="%.2f")
    r_pct = st.number_input("Risk-free rate r (%)", min_value=-20.0, max_value=100.0, value=5.0, step=0.25, format="%.2f")
    vol_pct = st.number_input("Volatility σ (%)", min_value=0.1, max_value=300.0, value=20.0, step=1.0, format="%.2f")
    T = st.number_input("Maturity T (years)", min_value=0.01, max_value=50.0, value=1.0, step=0.25, format="%.2f")
    N = st.slider("Lattice steps N", min_value=1, max_value=50, value=5)
    max_n = st.slider("Convergence up to N", min_value=20, max_value=200, value=120, step=10)
    st.markdown("---")
    kind = st.radio("Option type", ["Call", "Put"], index=0, horizontal=True)
    style = st.radio("Exercise style", ["European", "American"], index=0, horizontal=True)
    st.markdown("---")
    st.caption("Pink rings mark non-terminal nodes where early exercise is optimal in the American model.")

r = r_pct / 100.0
sigma = vol_pct / 100.0

try:
    tree = price_tree(S0, K, r, sigma, T, N, kind, style)
    delta, gamma, vega, theta = greeks_finite_difference(S0, K, r, sigma, T, N, kind, style)
    bs_european = black_scholes(S0, K, r, sigma, T, kind)
    european_tree = european_price_fast(S0, K, r, sigma, T, N, kind)
except (ValueError, OverflowError, ZeroDivisionError) as exc:
    st.error(str(exc))
    st.stop()

# -------------------- PREMIUM SUMMARY --------------------
st.markdown("### Pricing summary")
price_gap = abs(european_tree - bs_european)
c1, c2, c3 = st.columns(3, gap="medium")
with c1:
    st.markdown(kpi_card(f"Binomial price (N = {N})", f"{tree['price']:.4f}", f"{style} {kind} · CRR model", BLUE), unsafe_allow_html=True)
with c2:
    st.markdown(kpi_card("Black–Scholes (European)", f"{bs_european:.4f}", "Continuous-time benchmark", ORANGE), unsafe_allow_html=True)
with c3:
    st.markdown(kpi_card("|European binomial − BS|", f"{price_gap:.4f}", f"Like-for-like convergence gap · N = {N}", "#d8e0ef"), unsafe_allow_html=True)

# Optional info to distinguish early exercise premium from European convergence gap.
if style == "American":
    premium = max(0.0, tree["price"] - european_tree)
    st.caption(f"American exercise premium over the European binomial price: {premium:.4f}. The convergence gap above compares European binomial with European Black–Scholes, not American with European pricing.")

st.markdown("### Option Greeks")
g1, g2, g3, g4 = st.columns(4, gap="medium")
with g1:
    st.markdown(kpi_card("Δ  Delta", f"{delta:.4f}", "Approx. price change per 1-unit spot move", BLUE), unsafe_allow_html=True)
with g2:
    st.markdown(kpi_card("Γ  Gamma", f"{gamma:.5f}", "Rate of change in delta", TEAL), unsafe_allow_html=True)
with g3:
    st.markdown(kpi_card("ν  Vega / 1%", f"{vega:.4f}", "Value change for +1 volatility point", PURPLE), unsafe_allow_html=True)
with g4:
    theta_text = f"{theta:.4f}" if math.isfinite(theta) else "—"
    st.markdown(kpi_card("Θ  Theta / day", theta_text, "Approx. calendar-day time decay", PINK), unsafe_allow_html=True)

st.caption("Greeks are finite-difference estimates from the selected binomial model. Vega is for a one percentage-point volatility change; theta is approximate calendar-day decay.")
st.divider()

# -------------------- LATTICE --------------------
st.markdown('<div class="section-intro"><b>01 · Recombining binomial lattice</b><p>Each node includes the underlying price and option value. Hover over a node for exercise and continuation values. The colour scale shows option value; pink rings identify early-exercise nodes.</p></div>', unsafe_allow_html=True)
st.plotly_chart(lattice_figure(tree, N, kind), use_container_width=True, config={"displaylogo": False, "responsive": True})

# -------------------- CONVERGENCE --------------------
st.markdown('<div class="section-intro"><b>02 · Convergence to Black–Scholes</b><p>The European CRR price is compared with the European Black–Scholes value. As the number of time steps increases, the binomial result generally converges toward the continuous-time benchmark, often with a small sawtooth pattern.</p></div>', unsafe_allow_html=True)
conv_fig, bs_check, ns, conv_vals = convergence_figure(S0, K, r, sigma, T, max_n, kind)
st.plotly_chart(conv_fig, use_container_width=True, config={"displaylogo": False, "responsive": True})
if ns:
    last_gap = abs(conv_vals[-1] - bs_check)
    x1, x2, x3 = st.columns(3)
    x1.metric(f"Last European binomial price (N = {ns[-1]})", f"{conv_vals[-1]:.5f}")
    x2.metric("Black–Scholes price", f"{bs_check:.5f}")
    x3.metric("Absolute convergence gap", f"{last_gap:.5f}")

st.divider()

# -------------------- RISK-NEUTRAL SUMMARY --------------------
st.markdown('<div class="section-intro"><b>03 · Risk-neutral intuition</b><p>Under the risk-neutral measure, the discounted expected terminal payoff gives the European binomial price. An American option also compares continuation value with immediate exercise at each eligible node.</p></div>', unsafe_allow_html=True)
try:
    dt, u, d, p, disc = crr_params(r, sigma, T, N)
    j = np.arange(N + 1)
    terminal_stock = S0 * (u ** j) * (d ** (N-j))
    probs = np.array([math.comb(N, int(k)) * (p ** k) * ((1-p) ** (N-k)) for k in j], dtype=float)
    expected_payoff = float(np.dot(probs, payoff(terminal_stock, K, kind)))
    discounted_expected = math.exp(-r*T) * expected_payoff
    q1, q2, q3, q4 = st.columns(4, gap="medium")
    q1.metric("Step Δt", f"{dt:.5f}")
    q2.metric("Up factor u", f"{u:.5f}")
    q3.metric("Down factor d", f"{d:.5f}")
    q4.metric("Risk-neutral probability p", f"{p:.5f}")
    q1, q2, q3, q4 = st.columns(4, gap="medium")
    q1.metric("Step discount e⁻ʳΔᵗ", f"{disc:.5f}")
    q2.metric("Total discount e⁻ʳᵀ", f"{math.exp(-r*T):.5f}")
    q3.metric("Eᴽ[payoff]", f"{expected_payoff:.4f}")
    q4.metric("Discounted Eᴽ", f"{discounted_expected:.4f}")
    if style == "European":
        st.success(f"Consistency check: European binomial price = {tree['price']:.4f}; discounted expected payoff = {discounted_expected:.4f} (equal up to numerical rounding).")
    else:
        st.info(f"Discounted terminal expectation ({discounted_expected:.4f}) is the European binomial value. The American value ({tree['price']:.4f}) also considers optimal early exercise.")
except (ValueError, OverflowError) as exc:
    st.warning(f"Risk-neutral summary unavailable: {exc}")

# -------------------- MATHEMATICS --------------------
with st.expander("The mathematics · formulas and interpretation", expanded=True):
    st.markdown('<div class="math-card"><h4>1. Terminal payoff</h4><p>The option payoff at expiry depends on whether the contract is a call or a put.</p></div>', unsafe_allow_html=True)
    if kind == "Call":
        st.latex(r"\Phi(S_T)=(S_T-K)^+=\max(S_T-K,0)")
    else:
        st.latex(r"\Phi(S_T)=(K-S_T)^+=\max(K-S_T,0)")

    st.markdown('<div class="math-card"><h4>2. Recombining stock-price tree</h4><p>At step i, j counts the number of up moves. The order of up/down moves does not change the resulting price when fixed CRR factors are used.</p></div>', unsafe_allow_html=True)
    st.latex(r"S_i^j=S_0u^jd^{\,i-j},\qquad 0\leq j\leq i\leq N")

    st.markdown('<div class="math-card"><h4>3. Cox–Ross–Rubinstein (CRR) parameters</h4><p>These factors match the chosen volatility in a discrete-time tree and define the risk-neutral up probability.</p></div>', unsafe_allow_html=True)
    st.latex(r"\Delta t=\frac{T}{N},\qquad u=e^{\sigma\sqrt{\Delta t}},\qquad d=e^{-\sigma\sqrt{\Delta t}}=\frac{1}{u}")
    st.latex(r"p=\frac{e^{r\Delta t}-d}{u-d},\qquad 1-p=\frac{u-e^{r\Delta t}}{u-d}")
    st.latex(r"d<e^{r\Delta t}<u\quad\Longleftrightarrow\quad 0<p<1")

    st.markdown('<div class="math-card"><h4>4. European option · backward induction</h4><p>Start from the terminal payoff and work backwards. At each node, the value is the discounted risk-neutral expected value at the next step.</p></div>', unsafe_allow_html=True)
    st.latex(r"V_N^j=\Phi(S_N^j)")
    st.latex(r"V_i^j=e^{-r\Delta t}\left[pV_{i+1}^{j+1}+(1-p)V_{i+1}^{j}\right]")

    st.markdown('<div class="math-card"><h4>5. American option · early exercise</h4><p>An American option compares immediate exercise value with continuation value at every eligible node and takes the greater value.</p></div>', unsafe_allow_html=True)
    st.latex(r"V_i^j=\max\left(\Phi(S_i^j),\ e^{-r\Delta t}\left[pV_{i+1}^{j+1}+(1-p)V_{i+1}^{j}\right]\right)")
    st.latex(r"\text{Early exercise is optimal when }\Phi(S_i^j)>\text{Continuation value}")

    st.markdown('<div class="math-card"><h4>6. Risk-neutral valuation</h4><p>For a European option, the price is the discounted expectation of its terminal payoff under the risk-neutral probability measure Q.</p></div>', unsafe_allow_html=True)
    st.latex(r"V_0=e^{-rT}\mathbb{E}^{\mathbb{Q}}[\Phi(S_T)]")
    st.latex(r"\mathbb{E}^{\mathbb{Q}}[S_{t+\Delta t}\mid S_t]=S_te^{r\Delta t}")

    st.markdown('<div class="math-card"><h4>7. Black–Scholes European benchmark</h4><p>The continuous-time reference price uses the same spot, strike, rate, volatility and maturity inputs.</p></div>', unsafe_allow_html=True)
    st.latex(r"d_1=\frac{\ln(S_0/K)+(r+\tfrac12\sigma^2)T}{\sigma\sqrt{T}},\qquad d_2=d_1-\sigma\sqrt{T}")
    st.latex(r"C=S_0\mathcal{N}(d_1)-Ke^{-rT}\mathcal{N}(d_2)")
    st.latex(r"P=Ke^{-rT}\mathcal{N}(-d_2)-S_0\mathcal{N}(-d_1)")

    st.markdown('<div class="math-card"><h4>8. Greeks · local sensitivities</h4><p>The displayed Greeks are estimated from the selected binomial model using finite differences.</p></div>', unsafe_allow_html=True)
    st.latex(r"\Delta\approx\frac{V(S_0+h)-V(S_0-h)}{2h}")
    st.latex(r"\Gamma\approx\frac{V(S_0+h)-2V(S_0)+V(S_0-h)}{h^2}")
    st.latex(r"\text{Vega}_{1\%}\approx\frac{V(\sigma+0.01)-V(\sigma-0.01)}{2}")
    st.latex(r"\Theta_{\text{day}}\approx V(T-1/365)-V(T)")

    st.markdown('<div class="math-card"><h4>9. Convergence gap</h4><p>The comparison is made European-to-European so the benchmark is consistent. CRR prices can oscillate around the Black–Scholes value as N changes.</p></div>', unsafe_allow_html=True)
    st.latex(r"\text{Gap}_N=\left|V_{0,\,\mathrm{European}}^{\mathrm{CRR}}(N)-V_{\mathrm{BS}}\right|")

st.caption("Educational pricing model. Assumes constant volatility and risk-free rate, no dividends, and frictionless markets. Model estimates are not investment advice.")
