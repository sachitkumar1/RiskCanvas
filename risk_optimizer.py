from __future__ import annotations
"""
Risk Optimizer UI — Integrated with RL Model for specific buy/sell recommendations for stocks/options. Includes monte-carlo simulation feature
Uses Polygon.io API for real-time price data
"""
import json
import math
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import requests

# Fix for numpy compatibility issues with saved models
import sys
if 'numpy.core' not in sys.modules:
    try:
        import numpy.core._multiarray_umath
        import numpy.core._multiarray_tests
    except ImportError:
        pass

from kivy.clock import Clock
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.metrics import dp
from kivy.properties import (
    DictProperty,
    ListProperty,
    NumericProperty,
    ObjectProperty,
    StringProperty,
)
from kivy.uix.screenmanager import Screen, ScreenManager, NoTransition
from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import (
    MDRaisedButton,
    MDFlatButton,
    MDRectangleFlatButton,
    MDIconButton,
)
from kivymd.uix.card import MDCard
from kivymd.uix.dialog import MDDialog
from kivymd.uix.list import OneLineRightIconListItem, IconRightWidget, TwoLineListItem
from kivymd.toast import toast
from kivymd.uix.textfield import MDTextField
from matplotlib import pyplot as plt
from scipy.stats import norm

# RL Model imports
try:
    import gym
    from gym import spaces
    from stable_baselines3 import PPO
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    print("Warning: stable-baselines3 or gym not available. RL features disabled.")

# UI
KV = """
<Header@MDBoxLayout>:
    adaptive_height: True
    padding: dp(12)
    md_bg_color: app.theme_cls.primary_color
    MDIconButton:
        icon: root.icon if hasattr(root, 'icon') else 'chart-line'
        on_release: app.go_back()
    MDLabel:
        text: root.title if hasattr(root, 'title') else 'Risk Optimizer'
        font_style: 'H4'
        halign: 'left'
    MDIconButton:
        icon: 'logout'
        on_release: app.logout()

<Chip@MDCard>:
    radius: [16,16,16,16]
    padding: dp(10)
    md_bg_color: app.theme_cls.accent_color if hasattr(app.theme_cls, 'accent_color') else app.theme_cls.primary_color

<Divider@MDBoxLayout>:
    size_hint_y: None
    height: dp(1)
    md_bg_color: app.theme_cls.divider_color if hasattr(app.theme_cls, 'divider_color') else (0.2,0.2,0.2,1)

<AuthScreen>:
    name: 'auth'
    MDBoxLayout:
        orientation: 'vertical'
        spacing: dp(12)
        padding: dp(16)
        Header:
            title: 'Welcome'
            icon: 'login'
        MDCard:
            orientation: 'vertical'
            padding: dp(16)
            radius: [24,24,24,24]
            spacing: dp(8)
            MDTextField:
                id: username
                hint_text: 'Username'
                text: root.username
                on_text: root.username = self.text
            MDTextField:
                id: password
                hint_text: 'Password'
                password: True
                text: root.password
                on_text: root.password = self.text
            MDBoxLayout:
                spacing: dp(8)
                MDRaisedButton:
                    text: 'Log in'
                    on_release: root.login()
                MDRectangleFlatButton:
                    text: 'Register'
                    on_release: root.register()
                MDRectangleFlatButton:
                    text: 'Settings'
                    on_release: app.goto('settings')

<SettingsScreen>:
    name: 'settings'
    MDBoxLayout:
        orientation: 'vertical'
        spacing: dp(12)
        padding: dp(16)
        Header:
            title: 'Settings'
            icon: 'cog'
        MDCard:
            orientation: 'vertical'
            radius: [24,24,24,24]
            padding: dp(16)
            spacing: dp(8)
            MDTextField:
                id: api_key
                hint_text: 'Polygon.io API Key'
                text: root.api_key
                on_text: root.api_key = self.text
            MDRaisedButton:
                text: 'Save'
                on_release: root.save()

<QuestionnaireScreen>:
    name: 'questionnaire'
    MDBoxLayout:
        orientation: 'vertical'
        padding: dp(16)
        spacing: dp(12)
        Header:
            title: 'Risk Questionnaire'
            icon: 'clipboard-text'
        MDCard:
            id: qcard
            orientation: 'vertical'
            radius: [24,24,24,24]
            padding: dp(16)
            spacing: dp(8)
            MDLabel:
                id: qtitle
                text: root.current_question_title
                font_style: 'H5'
                adaptive_height: True
            Divider:
            MDBoxLayout:
                id: qoptions
                orientation: 'vertical'
                spacing: dp(8)
            Divider:
            MDBoxLayout:
                adaptive_height: True
                spacing: dp(8)
                MDRectangleFlatButton:
                    text: 'Back'
                    on_release: root.prev_question()
                MDRaisedButton:
                    text: 'Next'
                    on_release: root.next_question()

<PortfolioScreen>:
    name: 'portfolio'
    MDBoxLayout:
        orientation: 'vertical'
        padding: dp(16)
        spacing: dp(12)
        Header:
            title: 'My Portfolio'
            icon: 'briefcase-variant'
        MDCard:
            radius: [24,24,24,24]
            padding: dp(12)
            MDBoxLayout:
                orientation: 'vertical'
                spacing: dp(8)
                MDLabel:
                    text: 'Stocks / ETFs'
                    font_style: 'Subtitle1'
                MDBoxLayout:
                    spacing: dp(8)
                    MDTextField:
                        id: stock_ticker
                        hint_text: 'Ticker (e.g., AAPL)'
                    MDTextField:
                        id: stock_value
                        hint_text: 'Market Value ($)'
                        input_filter: 'float'
                    MDRaisedButton:
                        text: 'Add'
                        on_release: root.add_stock(stock_ticker.text, stock_value.text)
                MDScrollView:
                    size_hint_y: None
                    height: dp(200)
                    do_scroll_x: False
                    bar_width: dp(4)
                    MDList:
                        id: stock_list
                Divider:
                MDLabel:
                    text: 'Options'
                    font_style: 'Subtitle1'
                MDBoxLayout:
                    spacing: dp(8)
                    MDTextField:
                        id: opt_ticker
                        hint_text: 'Underlying (e.g., AAPL)'
                    MDTextField:
                        id: opt_type
                        hint_text: 'Type (C/P)'
                        max_text_length: 1
                    MDTextField:
                        id: opt_strike
                        hint_text: 'Strike'
                        input_filter: 'float'
                    MDTextField:
                        id: opt_expiry
                        hint_text: 'Expiry (YYYY-MM-DD)'
                    MDTextField:
                        id: opt_qty
                        hint_text: 'Contracts'
                        input_filter: 'int'
                    MDRaisedButton:
                        text: 'Add'
                        on_release: root.add_option(opt_ticker.text, opt_type.text, opt_strike.text, opt_expiry.text, opt_qty.text)
                MDScrollView:
                    size_hint_y: None
                    height: dp(200)
                    do_scroll_x: False
                    bar_width: dp(4)
                    MDList:
                        id: option_list
                MDBoxLayout:
                    adaptive_height: True
                    spacing: dp(8)
                    MDRaisedButton:
                        text: 'Get Optimization'
                        on_release: root.compute_risk()
                    MDRectangleFlatButton:
                        text: 'Run Monte Carlo'
                        on_release: root.run_monte_carlo()

<ResultsScreen>:
    name: 'results'
    MDBoxLayout:
        orientation: 'vertical'
        padding: dp(16)
        spacing: dp(12)
        Header:
            title: 'Results & Recommendations'
            icon: 'chart-areaspline'
        MDScrollView:
            do_scroll_x: False
            MDBoxLayout:
                orientation: 'vertical'
                spacing: dp(12)
                adaptive_height: True
                MDCard:
                    radius: [24,24,24,24]
                    padding: dp(16)
                    orientation: 'vertical'
                    adaptive_height: True
                    spacing: dp(8)
                    MDLabel:
                        text: f"Target Risk (V): {root.v_score:.2f} / 100"
                        font_style: 'H5'
                        adaptive_height: True
                    MDLabel:
                        text: f"Current Portfolio Risk (Z): {root.z_score:.2f} / 100"
                        font_style: 'H5'
                        adaptive_height: True
                    MDLabel:
                        text: f"Optimized Portfolio Risk: {root.optimized_z:.2f} / 100"
                        font_style: 'H5'
                        adaptive_height: True
                        theme_text_color: 'Primary'
                    Divider:
                    MDLabel:
                        text: 'Factor Scores'
                        font_style: 'Subtitle1'
                        adaptive_height: True
                    MDBoxLayout:
                        adaptive_height: True
                        spacing: dp(8)
                        Chip:
                            MDLabel:
                                text: f"Sigma: {root.sigma_score:.2f}"
                        Chip:
                            MDLabel:
                                text: f"Beta: {root.beta_score:.2f}"
                        Chip:
                            MDLabel:
                                text: f"VaR: {root.var_score:.2f}"
                        Chip:
                            MDLabel:
                                text: f"MDD: {root.mdd_score:.2f}"
                        Chip:
                            MDLabel:
                                text: f"Sharpe (inv): {root.sharpe_inv_score:.2f}"
                MDCard:
                    radius: [24,24,24,24]
                    padding: dp(16)
                    orientation: 'vertical'
                    adaptive_height: True
                    spacing: dp(8)
                    MDLabel:
                        text: 'Stock/ETF Recommendations'
                        font_style: 'H5'
                        adaptive_height: True
                    Divider:
                    MDList:
                        id: stock_recommendations
                MDCard:
                    radius: [24,24,24,24]
                    padding: dp(16)
                    orientation: 'vertical'
                    adaptive_height: True
                    spacing: dp(8)
                    MDLabel:
                        text: 'Options Recommendations'
                        font_style: 'H5'
                        adaptive_height: True
                    Divider:
                    MDList:
                        id: options_recommendations
                MDBoxLayout:
                    adaptive_height: True
                    spacing: dp(8)
                    MDRectangleFlatButton:
                        text: 'Back to Portfolio'
                        on_release: app.goto('portfolio')
"""

# Data storage for users
DATA_FILE = "risk_optimizer_data.json"

def load_store() -> dict:
    if not os.path.exists(DATA_FILE):
        return {"users": {}}
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_store(store: dict) -> None:
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2)

# Constants for the algorithm. W = weights in calculation
W_SIGMA = 0.20
W_BETA = 0.15
W_SHARPE = 0.20
W_VAR = 0.25
W_MDD = 0.20
W_CAPACITY = 0.60
W_TOLERANCE = 0.40
MAX_RAW_SCORE = 28
RISK_FREE_RATE_ANNUAL = 0.045
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = RISK_FREE_RATE_ANNUAL / math.sqrt(TRADING_DAYS_PER_YEAR)

NORMALIZATION_RANGES = {
    'sigma': {'min': 0.03, 'max': 0.50},
    'beta': {'min': 0.0, 'max': 2.5},
    'sharpe': {'min': 0.0, 'max': 4.0},
    'var': {'min': 0.005, 'max': 0.40},
    'mdd': {'min': 0.01, 'max': 0.60},
}

BASE_URL_AGGS = "https://api.polygon.io/v2/aggs/ticker"
DEFAULT_POLYGON_API_KEY = "3CV4EpFz71FVqW7tJK792uXv8WMWhnTX"
MARKET_INDEX_TICKER = "SPY"

API_WINDOW_MAX = 5
API_WINDOW_SECONDS = 63

# Asset order for RL model. Didn't have enough time to train the RL model on more assets because we had limited time (hackathon) and computing power. 
RL_ASSETS = ['SPY', 'AAPL', 'GOOGL', 'MSFT', 'META', 'NVDA']

CAPACITY_QS = {
    "C1: Investment Time Horizon (Age)": {
        "Over 20 years (e.g., age < 45)": 10,
        "10-20 years (e.g., age 45-55)": 6,
        "Less than 10 years (e.g., age > 55)": 2,
    },
    "C2: Liquidity Needs (Savings & Expenses)": {
        "I have enough savings to cover 12+ months of expenses.": 10,
        "I have enough savings to cover 6-12 months of expenses.": 5,
        "I have savings for less than 6 months of expenses.": 1,
    },
    "C3: Income Stability": {
        "My income is highly stable (e.g., government job, large company).": 8,
        "My income is moderately stable (e.g., established mid-size company).": 4,
        "My income is variable or at a startup (high instability).": 2,
    },
}

TOLERANCE_QS = {
    "T1: Reaction to a 20% Portfolio Loss": {
        "I'd buy more, as it's a buying opportunity.": 10,
        "I'd do nothing and wait for it to recover.": 6,
        "I'd sell everything to avoid further losses.": 2,
    },
    "T2: Investment Preference": {
        "Highest possible growth, even if it means high risk.": 10,
        "Moderate growth, accepting similar risk to the stock market.": 5,
        "Capital protection is my priority (safer portfolio).": 1,
    },
    "T3: Investment Knowledge": {
        "I have an excellent understanding of market risk and complex instruments.": 8,
        "I have a basic understanding of stocks and mutual funds.": 4,
        "I have little to no market knowledge.": 1,
    },
}

# Limits our program to 5 API calls per minute, since that is all that is allowed with the free version of our API access to polygon API.
class RateLimiter:
    def __init__(self, max_calls: int = API_WINDOW_MAX, window_seconds: int = API_WINDOW_SECONDS):
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self.window_start: float | None = None
        self.count = 0

    def wait(self):
        now = time.time()
        if self.window_start is None:
            self.window_start = now
            self.count = 0
        elapsed = now - self.window_start
        if self.count >= self.max_calls:
            if elapsed < self.window_seconds:
                time.sleep(self.window_seconds - elapsed)
            self.window_start = time.time()
            self.count = 0
        self.count += 1

# Market Data Extraction
def fetch_market_data(tickers: List[str], api_key: str, days: int = TRADING_DAYS_PER_YEAR) -> pd.DataFrame:
    to_date = datetime.now().strftime('%Y-%m-%d')
    from_date = (datetime.now() - timedelta(days=int(365 * 1.2))).strftime('%Y-%m-%d')
    data_store: Dict[str, pd.Series] = {}
    if MARKET_INDEX_TICKER not in tickers:
        tickers = list(tickers) + [MARKET_INDEX_TICKER]
    rl = RateLimiter()
    for ticker in tickers:
        url = (
            f"{BASE_URL_AGGS}/{ticker}/range/1/day/{from_date}/{to_date}?"
            f"adjusted=true&limit=50000&apiKey={api_key}"
        )
        try:
            rl.wait()
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200:
                time.sleep(1)
                continue
            j = resp.json()
            if (j.get('status') in ('OK', 'DELAYED')) and j.get('results'):
                prices: Dict[str, float] = {}
                for bar in j['results']:
                    date = datetime.fromtimestamp(bar['t'] / 1000).strftime('%Y-%m-%d')
                    prices[date] = bar['c']
                if prices:
                    s = pd.Series(prices).sort_index()
                    data_store[ticker] = s.tail(days)
            elif j.get('status') == 'NOT_FOUND':
                pass
        except requests.RequestException:
            time.sleep(1)
    price_data = pd.DataFrame(data_store)
    if MARKET_INDEX_TICKER in price_data.columns:
        price_data.rename(columns={MARKET_INDEX_TICKER: 'MARKET_INDEX'}, inplace=True)
    return price_data.dropna()

def load_returns_data():
    """Load returns data for RL model"""
    try:
        returns_df = pd.read_csv('returns_polygon.csv', index_col=0, parse_dates=True)
        column_mapping = {}
        for col in returns_df.columns:
            if col.startswith('Investment_'):
                column_mapping[col] = col.replace('Investment_', '')
        if column_mapping:
            returns_df = returns_df.rename(columns=column_mapping)
        return returns_df
    except FileNotFoundError:
        print("Warning: returns_polygon.csv not found")
        return None

# Black Scholes calculations for options
def black_scholes(S, K, T, r, sigma, option_type='C') -> Tuple[float, float]:
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type.upper() == 'C':
        delta = norm.cdf(d1)
        price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        delta = norm.cdf(d1) - 1
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return float(delta), float(price)

# calculating values (portfolio metrics)
def calculate_portfolio_metrics_with_options(portfolio_details: dict, prices: pd.DataFrame) -> Tuple[dict, pd.Series]:
    stocks: Dict[str, float] = portfolio_details.get('stocks', {})
    options: List[dict] = portfolio_details.get('options', [])
    delta_adj = stocks.copy()

    for opt in options:
        ticker = opt['ticker']
        if ticker not in prices.columns or prices[ticker].empty:
            continue
        S = float(prices[ticker].iloc[-1])
        if ticker in prices.columns and len(prices[ticker].dropna()) > 2:
            hist_vol = prices[ticker].pct_change().dropna().std() * math.sqrt(TRADING_DAYS_PER_YEAR)
        else:
            hist_vol = 0.30
        try:
            expiry_date = datetime.strptime(opt['expiry'], '%Y-%m-%d')
            T = max((expiry_date - datetime.now()).days / 365, 0.001)
        except Exception:
            T = 0.25
        K = float(opt['strike'])
        delta, price = black_scholes(S, K, T, RISK_FREE_RATE_ANNUAL, hist_vol, opt['type'])
        delta_value = abs(delta) * price * int(opt['quantity']) * 100
        delta_adj[ticker] = delta_adj.get(ticker, 0) + delta_value

    valid = [t for t in delta_adj.keys() if t in prices.columns]
    if not valid:
        return {'sigma': 0.0, 'beta': 0.0, 'sharpe': 0.0, 'var': 0.0, 'mdd': 0.0}, pd.Series(dtype=float)
    
    combined_val = sum(delta_adj[t] for t in valid)
    weights = np.array([delta_adj[t] / combined_val for t in valid])
    returns = prices.pct_change().dropna()
    
    if 'MARKET_INDEX' not in returns.columns:
        return {'sigma': 0.0, 'beta': 0.0, 'sharpe': 0.0, 'var': 0.0, 'mdd': 0.0}, pd.Series(dtype=float)
    
    market_returns = returns['MARKET_INDEX']
    if market_returns.var() == 0 or len(market_returns) < 2:
        return {'sigma': 0.0, 'beta': 0.0, 'sharpe': 0.0, 'var': 0.0, 'mdd': 0.0}, pd.Series(dtype=float)
    
    portfolio_daily_returns_risk = returns[valid].dot(weights)
    sigma = float(portfolio_daily_returns_risk.std() * math.sqrt(TRADING_DAYS_PER_YEAR))

    # Beta
    variance = float(market_returns.var())
    asset_betas: Dict[str, float] = {}
    for t in valid:
        covariance = float(returns[t].cov(market_returns))
        asset_betas[t] = covariance / max(variance, 1e-12)
    beta = sum(asset_betas[t] * (delta_adj[t] / combined_val) for t in valid)

    # VaR
    daily_var = -float(portfolio_daily_returns_risk.quantile(0.05))
    var_ann = daily_var * math.sqrt(TRADING_DAYS_PER_YEAR)

    # MDD
    cum = (1 + portfolio_daily_returns_risk).cumprod()
    peak = cum.expanding(min_periods=1).max()
    drawdown = (cum / peak) - 1
    mdd = abs(float(drawdown.min()))

    # Sharpe
    stock_tickers_only = [t for t in valid if t in stocks]
    if stock_tickers_only:
        stock_values = np.array([stocks.get(t, 0) for t in stock_tickers_only])
        stock_weights = stock_values / max(stock_values.sum(), 1e-12)
        daily_sharpe_returns = returns[stock_tickers_only].dot(stock_weights)
    else:
        daily_sharpe_returns = portfolio_daily_returns_risk
    
    annual_return = float(daily_sharpe_returns.mean() * TRADING_DAYS_PER_YEAR)
    annual_risk_free = RISK_FREE_RATE * math.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = (annual_return - annual_risk_free) / max(sigma, 1e-9)

    return {
        'sigma': sigma,
        'beta': float(beta),
        'sharpe': float(sharpe),
        'var': float(var_ann),
        'mdd': float(mdd),
    }, portfolio_daily_returns_risk

def normalize_metric(value: float, metric_name: str, invert: bool = False) -> float:
    try:
        min_v = NORMALIZATION_RANGES[metric_name]['min']
        max_v = NORMALIZATION_RANGES[metric_name]['max']
    except KeyError:
        return 50.0
    clipped = float(np.clip(value, min_v, max_v))
    norm01 = (clipped - min_v) / (max_v - min_v)
    if invert:
        norm01 = 1.0 - norm01
    return 1 + (norm01 * 99)

def calculate_z(metrics: dict) -> Tuple[float, float, float, float, float, float]:
    s_sigma = normalize_metric(metrics['sigma'], 'sigma')
    s_beta = normalize_metric(metrics['beta'], 'beta')
    s_sharpe_inv = normalize_metric(metrics['sharpe'], 'sharpe', invert=True)
    s_var = normalize_metric(metrics['var'], 'var')
    s_mdd = normalize_metric(metrics['mdd'], 'mdd')
    Z = (
        W_SIGMA * s_sigma
        + W_BETA * s_beta
        + W_SHARPE * s_sharpe_inv
        + W_VAR * s_var
        + W_MDD * s_mdd
    )
    return float(Z), s_sigma, s_beta, s_sharpe_inv, s_var, s_mdd

# RL Environment
if RL_AVAILABLE:
    class PortfolioEnv(gym.Env):
        def __init__(self, initial_portfolio, target_v, returns_df=None):
            super().__init__()
            self.assets = RL_ASSETS
            self.n_assets = len(self.assets)
            self.target_v = target_v
            self.initial_portfolio = np.array(initial_portfolio, dtype=np.float32)
            self.portfolio = self.initial_portfolio.copy()
            self.max_steps = 50
            self.current_step = 0
            self.initial_total_value = self.initial_portfolio.sum()
            self.returns_df = returns_df
            
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)
            self.observation_space = spaces.Box(low=-10, high=10, shape=(self.n_assets + 2,), dtype=np.float32)

        def reset(self):
            self.portfolio = self.initial_portfolio.copy()
            self.current_step = 0
            return self._get_observation()

        def _get_observation(self):
            total = self.portfolio.sum() + 1e-9
            normalized_portfolio = self.portfolio / total
            normalized_portfolio = np.nan_to_num(normalized_portfolio, nan=0.0, posinf=1.0, neginf=0.0)
            metrics = self._calculate_metrics(self.portfolio)
            current_z = self._calculate_z(metrics)
            obs = np.concatenate([normalized_portfolio, [self.target_v / 100.0, current_z / 100.0]])
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)
            obs = np.clip(obs, -10, 10)
            return obs.astype(np.float32)

        def _calculate_metrics(self, portfolio_values):
            if self.returns_df is None or len(self.returns_df) == 0:
                weights = portfolio_values / portfolio_values.sum()
                concentration = np.sum(weights ** 2)
                return {
                    'sigma': np.clip(0.15 + concentration * 0.2, 0.03, 0.50),
                    'beta': 1.0,
                    'sharpe': np.clip(0.8 - concentration * 0.3, 0.0, 4.0),
                    'var': np.clip(0.05 + concentration * 0.1, 0.005, 0.40),
                    'mdd': np.clip(0.10 + concentration * 0.15, 0.01, 0.60)
                }
            
            total_value = portfolio_values.sum()
            if total_value == 0:
                return {'sigma': 0.3, 'beta': 1.0, 'sharpe': 0.5, 'var': 0.1, 'mdd': 0.2}
            
            weights = portfolio_values / total_value
            available_tickers = [t for t in self.assets if t in self.returns_df.columns]
            
            if len(available_tickers) >= 3:
                portfolio_returns = pd.Series(0.0, index=self.returns_df.index)
                for i, ticker in enumerate(self.assets):
                    if ticker in self.returns_df.columns:
                        portfolio_returns += self.returns_df[ticker].fillna(0) * weights[i]
                
                portfolio_returns = portfolio_returns.replace(0, np.nan).dropna()
                
                if len(portfolio_returns) > 30 and portfolio_returns.std() > 1e-6:
                    daily_std = portfolio_returns.std()
                    sigma = np.clip(daily_std * np.sqrt(252), 0.03, 0.50)
                    
                    if 'SPY' in self.returns_df.columns:
                        market_returns = self.returns_df['SPY'].dropna()
                        aligned_portfolio, aligned_market = portfolio_returns.align(market_returns, join='inner')
                        if len(aligned_portfolio) > 30 and aligned_market.var() > 1e-6:
                            covariance = aligned_portfolio.cov(aligned_market)
                            beta = np.clip(covariance / aligned_market.var(), 0.0, 2.5)
                        else:
                            beta = 1.0
                    else:
                        beta = 1.0
                    
                    annual_return = portfolio_returns.mean() * 252
                    annual_rf = 0.045
                    sharpe = np.clip((annual_return - annual_rf) / (sigma + 1e-9), 0.0, 4.0)
                    var_daily = abs(portfolio_returns.quantile(0.05))
                    var = np.clip(var_daily * np.sqrt(252), 0.005, 0.40)
                    cumulative = (1 + portfolio_returns).cumprod()
                    running_max = cumulative.expanding(min_periods=1).max()
                    drawdown = (cumulative / running_max) - 1
                    mdd = np.clip(abs(drawdown.min()), 0.01, 0.60)
                    
                    return {'sigma': sigma, 'beta': beta, 'sharpe': sharpe, 'var': var, 'mdd': mdd}
            
            weights = portfolio_values / portfolio_values.sum()
            concentration = np.sum(weights ** 2)
            return {
                'sigma': np.clip(0.15 + concentration * 0.2, 0.03, 0.50),
                'beta': 1.0,
                'sharpe': np.clip(0.8 - concentration * 0.3, 0.0, 4.0),
                'var': np.clip(0.05 + concentration * 0.1, 0.005, 0.40),
                'mdd': np.clip(0.10 + concentration * 0.15, 0.01, 0.60)
            }

        def _calculate_z(self, metrics):
            s_sigma = normalize_metric(metrics['sigma'], 'sigma')
            s_beta = normalize_metric(metrics['beta'], 'beta')
            s_sharpe_inv = normalize_metric(metrics['sharpe'], 'sharpe', invert=True)
            s_var = normalize_metric(metrics['var'], 'var')
            s_mdd = normalize_metric(metrics['mdd'], 'mdd')
            return W_SIGMA * s_sigma + W_BETA * s_beta + W_SHARPE * s_sharpe_inv + W_VAR * s_var + W_MDD * s_mdd

        def step(self, action):
            MAX_MOVE_DOLLAR = self.initial_total_value * 0.10
            scaled_action = action * MAX_MOVE_DOLLAR / self.max_steps
            action_magnitude = np.sum(np.abs(scaled_action))
            
            for i in range(self.n_assets):
                if scaled_action[i] < 0:
                    max_sell = self.portfolio[i]
                    scaled_action[i] = max(scaled_action[i], -max_sell)
            
            new_portfolio = self.portfolio + scaled_action
            new_portfolio = np.maximum(new_portfolio, 0)
            
            initial_total = self.initial_total_value
            new_total = new_portfolio.sum()
            if new_total > initial_total:
                scale_factor = initial_total / new_total
                new_portfolio = new_portfolio * scale_factor
            
            sell_out_penalty = 0
            for i in range(self.n_assets):
                if self.initial_portfolio[i] > 10 and new_portfolio[i] < 1:
                    sell_out_penalty += 10
            
            self.portfolio = new_portfolio
            metrics = self._calculate_metrics(self.portfolio)
            portfolio_z = self._calculate_z(metrics)
            
            diff = abs(self.target_v - portfolio_z)
            reward_z = -diff / 100.0
            ACTION_COST_FACTOR = 0.0001
            reward_action_cost = -action_magnitude * ACTION_COST_FACTOR
            reward = reward_z + reward_action_cost - sell_out_penalty
            
            self.current_step += 1
            done = self.current_step >= self.max_steps
            return self._get_observation(), reward, done, {}

# Options recommendations
def rebalance_options(current_calls, current_puts, target_amount):
    """Generate options trading recommendations based on target adjustment"""
    total_calls_value = sum(current_calls)
    total_puts_value = sum(current_puts)
    total_portfolio = total_calls_value + total_puts_value
    
    recommendation = {
        "close_calls": 0,
        "close_puts": 0,
        "open_calls": 0,
        "open_puts": 0,
        "net_change": 0,
        "details": []
    }
    
    if total_portfolio == 0:
        recommendation["details"].append("No existing options to trade")
        return recommendation
    
    # Need MORE bullish exposure (target_amount > 0)
    if target_amount > 0:
        available_capital = 0
        puts_to_close = 0
        for put_value in sorted(current_puts, reverse=True):
            if available_capital >= target_amount:
                break
            available_capital += put_value
            puts_to_close += 1
            recommendation["details"].append(f"Close PUT worth ${put_value:.2f}")
        
        recommendation["close_puts"] = puts_to_close
        avg_call_cost = total_calls_value / len(current_calls) if current_calls else 200
        calls_to_open = int(available_capital / avg_call_cost)
        recommendation["open_calls"] = calls_to_open
        recommendation["net_change"] = calls_to_open * avg_call_cost - available_capital
        
        for i in range(calls_to_open):
            recommendation["details"].append(f"Open CALL (~${avg_call_cost:.2f} each)")
    
    # Need MORE bearish exposure (target_amount < 0)
    elif target_amount < 0:
        target_reduction = abs(target_amount)
        available_capital = 0
        calls_to_close = 0
        for call_value in sorted(current_calls, reverse=True):
            if available_capital >= target_reduction:
                break
            available_capital += call_value
            calls_to_close += 1
            recommendation["details"].append(f"Close CALL worth ${call_value:.2f}")
        
        recommendation["close_calls"] = calls_to_close
        avg_put_cost = total_puts_value / len(current_puts) if current_puts else 150
        puts_to_open = int(available_capital / avg_put_cost)
        recommendation["open_puts"] = puts_to_open
        recommendation["net_change"] = -(puts_to_open * avg_put_cost)
        
        for i in range(puts_to_open):
            recommendation["details"].append(f"Open PUT (~${avg_put_cost:.2f} each)")
    
    # No change needed
    else:
        recommendation["details"].append("Portfolio is balanced - no trades needed")
    
    return recommendation

# RL Optimization function
def optimize_portfolio_with_rl(model, questions, current_portfolio, returns_df=None):
    """Run RL model to optimize portfolio"""
    # Calculate target V from questionnaire
    capacity_score = (questions['C1'] + questions['C2'] + questions['C3']) / 9.0
    tolerance_score = (questions['T1'] + questions['T2'] + questions['T3']) / 9.0
    target_v = (W_CAPACITY * capacity_score + W_TOLERANCE * tolerance_score) * 100
    
    # Build portfolio array in RL_ASSETS order
    portfolio_array = np.array([current_portfolio.get(asset, 0.0) for asset in RL_ASSETS])
    
    # Create environment
    env = PortfolioEnv(portfolio_array, target_v, returns_df)
    obs = env.reset()
    
    # Run optimization
    for step in range(env.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        if done:
            break
    
    final_portfolio = env.portfolio
    return final_portfolio, portfolio_array, target_v

# Monte Carlo Feature
def run_monte_carlo(
    initial_value: float,
    daily_returns: pd.Series,
    sigma: float,
    num_sims: int = 1000,
    num_days: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    if sigma <= 1e-9 or daily_returns.empty:
        return pd.DataFrame()
    daily_stdev = sigma / math.sqrt(TRADING_DAYS_PER_YEAR)
    daily_mean = float(daily_returns.mean())
    sims = np.zeros((num_days + 1, num_sims))
    for i in range(num_sims):
        dr = np.random.normal(loc=daily_mean, scale=daily_stdev, size=num_days)
        path = initial_value * np.cumprod(1 + dr)
        sims[:, i] = np.concatenate(([initial_value], path))
    return pd.DataFrame(sims)

def plot_mcs(sim_data: pd.DataFrame, initial_value: float, num_sims: int, num_days: int):
    if sim_data.empty:
        return
    plt.figure(figsize=(12, 7))
    plt.plot(sim_data, alpha=0.06)
    mean_path = sim_data.mean(axis=1)
    p95 = sim_data.quantile(0.95, axis=1)
    p05 = sim_data.quantile(0.05, axis=1)
    plt.plot(mean_path, linestyle='--', label='Mean Path')
    plt.plot(p95, label='95th Percentile')
    plt.plot(p05, label='5th Percentile')
    finals = sim_data.iloc[-1]
    title = f'Monte Carlo Simulation ({num_sims} Paths) - {num_days} Trading Days'
    plt.title(title)
    plt.xlabel('Trading Days')
    plt.ylabel('Portfolio Value (USD)')
    plt.grid(True, alpha=0.4, linestyle=':')
    plt.legend()
    txt = (
        f"Initial Value: ${initial_value:,.2f}\n"
        f"Median Final: ${finals.median():,.2f}\n"
        f"95th Pct Final: ${p95.iloc[-1]:,.2f}\n"
        f"5th Pct Final: ${p05.iloc[-1]:,.2f}\n"
        f"Best Final: ${finals.max():,.2f}\n"
        f"Worst Final: ${finals.min():,.2f}"
    )
    plt.text(0.02, 0.98, txt, transform=plt.gca().transAxes, va='top',
             bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    plt.show()

# Screens
class AuthScreen(Screen):
    username = StringProperty("")
    password = StringProperty("")

    def _toast(self, text: str):
        toast(text)

    def login(self):
        store = load_store()
        user = store['users'].get(self.username)
        if not user or user.get('password') != self.password:
            self._toast("Invalid credentials")
            return
        app = MDApp.get_running_app()
        app.current_user = self.username
        self._toast(f"Welcome back, {self.username}")
        app.goto('questionnaire')

    def register(self):
        if not self.username or not self.password:
            self._toast("Enter a username and password")
            return
        store = load_store()
        if self.username in store['users']:
            self._toast("User already exists")
            return
        store['users'][self.username] = {
            'password': self.password,
            'settings': {"api_key": ""},
            'questionnaire': {},
            'portfolio': {'stocks': {}, 'options': []},
        }
        save_store(store)
        self._toast("Registered! Please log in.")

class SettingsScreen(Screen):
    api_key = StringProperty("")

    def on_pre_enter(self, *args):
        app = MDApp.get_running_app()
        store = load_store()
        user = store['users'].get(app.current_user)
        if user:
            saved_key = user.get('settings', {}).get('api_key', '').strip()
            self.api_key = saved_key if saved_key else DEFAULT_POLYGON_API_KEY
        else:
            self.api_key = DEFAULT_POLYGON_API_KEY

    def save(self):
        app = MDApp.get_running_app()
        store = load_store()
        user = store['users'].get(app.current_user)
        if not user:
            toast("Please log in first")
            return
        user['settings']['api_key'] = self.api_key.strip()
        save_store(store)
        toast("Saved API key")

class QuestionnaireScreen(Screen):
    questions = ListProperty([])
    answers = DictProperty({})
    index = NumericProperty(0)
    current_question_title = StringProperty("")

    def on_pre_enter(self, *args):
        self.questions = []
        for section, block in (("capacity", CAPACITY_QS), ("tolerance", TOLERANCE_QS)):
            for q, opts in block.items():
                self.questions.append((section, q, opts))
        self.index = 0
        self.load_saved()
        self.render()

    def load_saved(self):
        app = MDApp.get_running_app()
        store = load_store()
        self.answers = store['users'].get(app.current_user, {}).get('questionnaire', {}) or {}

    def save_answers(self):
        app = MDApp.get_running_app()
        store = load_store()
        user = store['users'].get(app.current_user)
        if user is not None:
            user['questionnaire'] = self.answers
            save_store(store)

    def render(self):
        if not self.questions:
            return
        section, q, opts = self.questions[self.index]
        self.current_question_title = q
        qbox = self.ids.qoptions
        qbox.clear_widgets()
        
        for text, score in opts.items():
            btn = MDRaisedButton(text=text)
            def _on_release(b, section=section, q=q, score=score):
                self.answers[q] = score
                self.save_answers()
                toast("Selected")
            btn.bind(on_release=_on_release)
            qbox.add_widget(btn)

    def next_question(self):
        if self.index < len(self.questions) - 1:
            self.index += 1
            self.render()
        else:
            toast("Questionnaire completed")
            MDApp.get_running_app().goto('portfolio')

    def prev_question(self):
        if self.index > 0:
            self.index -= 1
            self.render()

    @staticmethod
    def compute_V(answers: Dict[str, int]) -> Tuple[float, int, int]:
        raw_capacity = sum(answers.get(q, 0) for q in CAPACITY_QS.keys())
        raw_tolerance = sum(answers.get(q, 0) for q in TOLERANCE_QS.keys())
        norm_capacity = (raw_capacity / MAX_RAW_SCORE) * 100
        norm_tolerance = (raw_tolerance / MAX_RAW_SCORE) * 100
        V = (W_CAPACITY * norm_capacity) + (W_TOLERANCE * norm_tolerance)
        return float(V), raw_capacity, raw_tolerance

class PortfolioScreen(Screen):
    def on_pre_enter(self, *args):
        self.refresh_lists()

    def _get_user(self):
        app = MDApp.get_running_app()
        store = load_store()
        return store, store['users'].get(app.current_user)

    def refresh_lists(self):
        store, user = self._get_user()
        if not user:
            return
        portfolio = user['portfolio']
        stocks_box = self.ids.stock_list
        opts_box = self.ids.option_list
        stocks_box.clear_widgets()
        opts_box.clear_widgets()
        
        for tkr, val in portfolio['stocks'].items():
            row = OneLineRightIconListItem(text=f"{tkr} — ${val:,.2f}")
            row.add_widget(IconRightWidget(icon='delete', on_release=lambda x, t=tkr: self.remove_stock(t)))
            stocks_box.add_widget(row)
        
        for i, opt in enumerate(portfolio['options']):
            row = OneLineRightIconListItem(text=f"{opt['ticker']} {opt['type']}{opt['strike']} {opt['expiry']} ×{opt['quantity']}")
            row.add_widget(IconRightWidget(icon='delete', on_release=lambda x, idx=i: self.remove_option(idx)))
            opts_box.add_widget(row)

    def add_stock(self, ticker: str, value: str):
        if not ticker or not value:
            toast("Enter ticker and value")
            return
        try:
            val = float(value)
        except ValueError:
            toast("Invalid value")
            return
        store, user = self._get_user()
        if not user:
            return
        user['portfolio']['stocks'][ticker.upper()] = val
        save_store(store)
        self.ids.stock_ticker.text = ''
        self.ids.stock_value.text = ''
        self.refresh_lists()

    def remove_stock(self, ticker: str):
        store, user = self._get_user()
        if not user:
            return
        user['portfolio']['stocks'].pop(ticker, None)
        save_store(store)
        self.refresh_lists()

    def add_option(self, tkr: str, typ: str, strike: str, expiry: str, qty: str):
        if not (tkr and typ and strike and expiry and qty):
            toast("Fill all option fields")
            return
        try:
            strike_f = float(strike)
            qty_i = int(qty)
        except ValueError:
            toast("Invalid strike/quantity")
            return
        typ = typ.upper()
        if typ not in ('C', 'P'):
            toast("Type must be C or P")
            return
        try:
            datetime.strptime(expiry, '%Y-%m-%d')
        except Exception:
            toast("Expiry must be YYYY-MM-DD")
            return
        store, user = self._get_user()
        if not user:
            return
        user['portfolio']['options'].append({
            'ticker': tkr.upper(), 'type': typ, 'strike': strike_f,
            'expiry': expiry, 'quantity': qty_i,
        })
        save_store(store)
        self.ids.opt_ticker.text = ''
        self.ids.opt_type.text = ''
        self.ids.opt_strike.text = ''
        self.ids.opt_expiry.text = ''
        self.ids.opt_qty.text = ''
        self.refresh_lists()

    def remove_option(self, idx: int):
        store, user = self._get_user()
        if not user:
            return
        if 0 <= idx < len(user['portfolio']['options']):
            user['portfolio']['options'].pop(idx)
            save_store(store)
        self.refresh_lists()

    def compute_risk(self):
        if getattr(self, "_computing", False):
            toast("Already computing…")
            return
        self._computing = True
        
        app = MDApp.get_running_app()
        store, user = self._get_user()
        if not user:
            self._computing = False
            return
        
        # Check if RL model is available
        if not RL_AVAILABLE:
            toast("RL libraries not installed. Install stable-baselines3 and gym.")
            self._computing = False
            return
        
        # Load RL model
        try:
            # Fix numpy compatibility issue
            import numpy as np
            
            # Try loading with different possible names
            if os.path.exists("final_model.zip"):
                try:
                    model = PPO.load("final_model")
                except Exception as load_error:
                    # If there's a numpy compatibility issue, try retraining or provide helpful message
                    if "numpy" in str(load_error).lower():
                        toast("Model has numpy compatibility issue. Please retrain the model with current numpy version.")
                    else:
                        toast(f"Model load error: {str(load_error)}")
                    self._computing = False
                    return
            elif os.path.exists("final_model"):
                model = PPO.load("final_model")
            else:
                toast("RL model (final_model.zip) not found in project directory")
                self._computing = False
                return
        except Exception as e:
            toast(f"Could not load RL model: {str(e)}")
            self._computing = False
            return
        
        # Load returns data for RL model
        returns_df = load_returns_data()
        
        # Get API key
        api_key = user.get('settings', {}).get('api_key', '').strip()
        if not api_key:
            api_key = DEFAULT_POLYGON_API_KEY
        
        portfolio = user['portfolio']
        if not portfolio['stocks'] and not portfolio['options']:
            toast("Add positions first")
            self._computing = False
            return
        
        # Fetch price data
        stock_tickers = set(portfolio['stocks'].keys())
        underlying_tickers = set(o['ticker'] for o in portfolio['options'])
        to_fetch = list(stock_tickers.union(underlying_tickers))
        
        prices = fetch_market_data(to_fetch, api_key)
        if prices.empty or 'MARKET_INDEX' not in prices.columns:
            toast("Could not fetch price data")
            self._computing = False
            return
        
        # Calculate current metrics
        metrics, daily_risk = calculate_portfolio_metrics_with_options(portfolio, prices)
        Z, s_sig, s_beta, s_shinv, s_var, s_mdd = calculate_z(metrics)
        
        # Get questionnaire answers and convert to numeric format
        answers = user.get('questionnaire', {})
        question_mapping = {
            'C1: Investment Time Horizon (Age)': {'C1': [10, 6, 2]},
            'C2: Liquidity Needs (Savings & Expenses)': {'C2': [10, 5, 1]},
            'C3: Income Stability': {'C3': [8, 4, 2]},
            'T1: Reaction to a 20% Portfolio Loss': {'T1': [10, 6, 2]},
            'T2: Investment Preference': {'T2': [10, 5, 1]},
            'T3: Investment Knowledge': {'T3': [8, 4, 1]},
        }
        
        numeric_questions = {}
        for q_text, score in answers.items():
            for key, q_dict in question_mapping.items():
                if key == q_text:
                    q_key = list(q_dict.keys())[0]
                    scores = q_dict[q_key]
                    if score == scores[0]:
                        numeric_questions[q_key] = 3
                    elif score == scores[1]:
                        numeric_questions[q_key] = 2
                    else:
                        numeric_questions[q_key] = 1
        
        # Ensure all questions have values
        for key in ['C1', 'C2', 'C3', 'T1', 'T2', 'T3']:
            if key not in numeric_questions:
                numeric_questions[key] = 2  # Default middle value
        
        # Build current portfolio dict for RL
        current_portfolio_dict = {}
        for asset in RL_ASSETS:
            current_portfolio_dict[asset] = portfolio['stocks'].get(asset, 0.0)
        
        # Run RL optimization
        try:
            final_portfolio, initial_array, target_v = optimize_portfolio_with_rl(
                model, numeric_questions, current_portfolio_dict, returns_df
            )
        except Exception as e:
            toast(f"Optimization failed: {str(e)}")
            self._computing = False
            return
        
        # Calculate optimized metrics
        optimized_stocks = {RL_ASSETS[i]: final_portfolio[i] for i in range(len(RL_ASSETS))}
        optimized_portfolio = {'stocks': optimized_stocks, 'options': portfolio['options']}
        opt_metrics, opt_daily = calculate_portfolio_metrics_with_options(optimized_portfolio, prices)
        opt_Z, _, _, _, _, _ = calculate_z(opt_metrics)
        
        # Generate stock recommendations
        stock_recommendations = []
        for i, asset in enumerate(RL_ASSETS):
            current = initial_array[i]
            recommended = final_portfolio[i]
            change = recommended - current
            
            if abs(change) > 10:
                action = "BUY" if change > 0 else "SELL"
                pct_change = (change / current * 100) if current > 0 else (100 if change > 0 else 0)
                stock_recommendations.append({
                    'asset': asset,
                    'action': action,
                    'change': abs(change),
                    'pct_change': pct_change,
                    'current': current,
                    'new': recommended
                })
            else:
                stock_recommendations.append({
                    'asset': asset,
                    'action': 'HOLD',
                    'change': 0,
                    'pct_change': 0,
                    'current': current,
                    'new': recommended
                })
        
        # Generate options recommendations by ticker
        options_recommendations = {}
        for i, asset in enumerate(RL_ASSETS):
            change = final_portfolio[i] - initial_array[i]
            
            # Get calls and puts for this ticker
            calls = [opt for opt in portfolio['options'] if opt['ticker'] == asset and opt['type'] == 'C']
            puts = [opt for opt in portfolio['options'] if opt['ticker'] == asset and opt['type'] == 'P']
            
            if calls or puts:
                # Calculate option values (simplified - using quantity * 100 as proxy)
                call_values = [opt['quantity'] * 100 for opt in calls]
                put_values = [opt['quantity'] * 100 for opt in puts]
                
                opt_rec = rebalance_options(call_values, put_values, change)
                options_recommendations[asset] = opt_rec
        
        # Store session data
        app.session = {
            'V': target_v,
            'Z': Z,
            'optimized_z': opt_Z,
            'scores': {
                'sigma': s_sig,
                'beta': s_beta,
                'var': s_var,
                'mdd': s_mdd,
                'sharpe_inv': s_shinv,
            },
            'metrics': metrics,
            'daily': daily_risk,
            'initial_value': float(sum(portfolio['stocks'].values())) or 10000.0,
            'stock_recommendations': stock_recommendations,
            'options_recommendations': options_recommendations,
        }
        
        self._computing = False
        app.goto('results')

    def run_monte_carlo(self):
        app = MDApp.get_running_app()
        sess = getattr(app, 'session', None)
        if not sess:
            toast("Compute risk first")
            return
        try:
            plt.close('all')
        except Exception:
            pass
        sim = run_monte_carlo(
            sess['initial_value'],
            sess['daily'],
            sess['metrics']['sigma'],
            1000,
            TRADING_DAYS_PER_YEAR,
        )
        if sim.empty:
            toast("Simulation could not run (need volatility & returns)")
            return
        plot_mcs(sim, sess['initial_value'], 1000, TRADING_DAYS_PER_YEAR)

class ResultsScreen(Screen):
    v_score = NumericProperty(0.0)
    z_score = NumericProperty(0.0)
    optimized_z = NumericProperty(0.0)
    sigma_score = NumericProperty(0.0)
    beta_score = NumericProperty(0.0)
    var_score = NumericProperty(0.0)
    mdd_score = NumericProperty(0.0)
    sharpe_inv_score = NumericProperty(0.0)

    def on_pre_enter(self, *args):
        app = MDApp.get_running_app()
        sess = getattr(app, 'session', {})
        self.v_score = float(sess.get('V', 0.0))
        self.z_score = float(sess.get('Z', 0.0))
        self.optimized_z = float(sess.get('optimized_z', 0.0))
        sc = sess.get('scores', {})
        self.sigma_score = float(sc.get('sigma', 0.0))
        self.beta_score = float(sc.get('beta', 0.0))
        self.var_score = float(sc.get('var', 0.0))
        self.mdd_score = float(sc.get('mdd', 0.0))
        self.sharpe_inv_score = float(sc.get('sharpe_inv', 0.0))
        
        # Display stock recommendations
        stock_box = self.ids.stock_recommendations
        stock_box.clear_widgets()
        for rec in sess.get('stock_recommendations', []):
            if rec['action'] == 'HOLD':
                item = TwoLineListItem(
                    text=f"{rec['asset']}: HOLD",
                    secondary_text=f"Current: ${rec['current']:,.2f}"
                )
            else:
                item = TwoLineListItem(
                    text=f"{rec['asset']}: {rec['action']} ${rec['change']:,.2f} ({rec['pct_change']:+.1f}%)",
                    secondary_text=f"Current: ${rec['current']:,.2f} → New: ${rec['new']:,.2f}"
                )
            stock_box.add_widget(item)
        
        # Display options recommendations
        opts_box = self.ids.options_recommendations
        opts_box.clear_widgets()
        for ticker, rec in sess.get('options_recommendations', {}).items():
            if rec['details']:
                summary = f"{ticker}: Close {rec['close_calls']} calls, {rec['close_puts']} puts | Open {rec['open_calls']} calls, {rec['open_puts']} puts"
                details = " | ".join(rec['details'][:3])  # Show first 3 details
                item = TwoLineListItem(
                    text=summary,
                    secondary_text=details
                )
                opts_box.add_widget(item)

# App (front end)
class RiskOptimizerApp(MDApp):
    current_user: str | None = None
    session: dict | None = None

    def build(self):
        self.title = "Risk Optimizer UI"
        self.theme_cls.primary_palette = "Blue"
        try:
            self.theme_cls.material_style = "M3"
        except Exception:
            pass
        root = ScreenManager(transition=NoTransition())
        Builder.load_string(KV)
        root.add_widget(AuthScreen())
        root.add_widget(SettingsScreen())
        root.add_widget(QuestionnaireScreen())
        root.add_widget(PortfolioScreen())
        root.add_widget(ResultsScreen())
        return root

    def goto(self, name: str):
        self.root.current = name

    def logout(self):
        self.current_user = None
        self.session = None
        try:
            toast("Logged out")
        except Exception:
            pass
        self.goto('auth')

    def go_back(self):
        if self.root.current == 'settings':
            self.goto('auth')
        elif self.root.current == 'questionnaire':
            self.goto('auth')
        elif self.root.current == 'portfolio':
            self.goto('questionnaire')
        elif self.root.current == 'results':
            self.goto('portfolio')

    def go_home(self):
        self.goto('auth')

if __name__ == "__main__":
    RiskOptimizerApp().run()
