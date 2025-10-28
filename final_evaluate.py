import gym
from gym import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
# from final_train import PortfolioEnv
W_SIGMA = 0.20
W_BETA = 0.15
W_SHARPE = 0.20
W_VAR = 0.25
W_MDD = 0.20

NORMALIZATION_RANGES = {
    'sigma': {'min': 0.03, 'max': 0.50},
    'beta': {'min': 0.0, 'max': 2.5},
    'sharpe': {'min': 0.0, 'max': 4.0},
    'var': {'min': 0.005, 'max': 0.40},
    'mdd': {'min': 0.01, 'max': 0.60}
}

def normalize_metric(value, metric_name, invert=False):
    min_val = NORMALIZATION_RANGES[metric_name]['min']
    max_val = NORMALIZATION_RANGES[metric_name]['max']
    clipped_value = np.clip(value, min_val, max_val)
    normalized = (clipped_value - min_val) / (max_val - min_val)
    if invert:
        normalized = 1 - normalized
    return 1 + normalized * 99

def calculate_z(metrics):
    score_sigma = normalize_metric(metrics['sigma'], 'sigma')
    score_beta = normalize_metric(metrics['beta'], 'beta')
    score_sharpe = normalize_metric(metrics['sharpe'], 'sharpe', invert=True)
    score_var = normalize_metric(metrics['var'], 'var')
    score_mdd = normalize_metric(metrics['mdd'], 'mdd')
    Z = (W_SIGMA * score_sigma + W_BETA * score_beta + W_SHARPE * score_sharpe +
         W_VAR * score_var + W_MDD * score_mdd)
    return Z

PRICES_DF = None
RETURNS_DF = None

def load_market_data():
    global PRICES_DF, RETURNS_DF
    try:
        RETURNS_DF = pd.read_csv('returns_polygon.csv', index_col=0, parse_dates=True)
        column_mapping = {}
        for col in RETURNS_DF.columns:
            if col.startswith('Investment_'):
                column_mapping[col] = col.replace('Investment_', '')
        
        if column_mapping:
            RETURNS_DF = RETURNS_DF.rename(columns=column_mapping)
            print(f"Renamed columns: {list(column_mapping.keys())}")
        
        try:
            PRICES_DF = pd.read_csv('prices_polygon.csv', index_col=0, parse_dates=True)
            if column_mapping:
                PRICES_DF = PRICES_DF.rename(columns=column_mapping)
        except FileNotFoundError:
            print("prices_polygon.csv not found (optional)")
            PRICES_DF = None
        
        print(f"Loaded market data: {len(RETURNS_DF)} days, Columns: {list(RETURNS_DF.columns)}")
        print(f"Date range: {RETURNS_DF.index[0]} to {RETURNS_DF.index[-1]}")
        
    except FileNotFoundError:
        print("Warning: returns_polygon.csv not found. Using synthetic data.")
        RETURNS_DF = None
        PRICES_DF = None

def calculate_portfolio_metrics(portfolio_values):
    global RETURNS_DF
    
    portfolio_values = np.array(portfolio_values)
    
    if len(portfolio_values) < 2 or np.sum(portfolio_values) == 0:
        return {'sigma': 0.3, 'beta': 1.0, 'sharpe': 0.5, 'var': 0.1, 'mdd': 0.2}
    
    if RETURNS_DF is None:
        load_market_data()
    
    if RETURNS_DF is not None and len(RETURNS_DF) > 0:
        tickers = ['SPY','AAPL', 'GOOGL', 'MSFT', 'META', 'NVDA']
        total_value = portfolio_values.sum()
        if total_value == 0:
            return {'sigma': 0.3, 'beta': 1.0, 'sharpe': 0.5, 'var': 0.1, 'mdd': 0.2}
        
        weights = portfolio_values / total_value
        
        available_tickers = [t for t in tickers if t in RETURNS_DF.columns]
        
        if len(available_tickers) >= 3:  
            portfolio_returns = pd.Series(0.0, index=RETURNS_DF.index)
            
            for i, ticker in enumerate(tickers):
                if ticker in RETURNS_DF.columns:
                    portfolio_returns += RETURNS_DF[ticker].fillna(0) * weights[i]
            
            portfolio_returns = portfolio_returns.replace(0, np.nan).dropna()
            
            if len(portfolio_returns) > 30 and portfolio_returns.std() > 1e-6:
                daily_std = portfolio_returns.std()
                sigma = np.clip(daily_std * np.sqrt(252), 0.03, 0.50)
                
                if 'SPY' in RETURNS_DF.columns:
                    market_returns = RETURNS_DF['SPY'].dropna()
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
                return {
                    'sigma': sigma,
                    'beta': beta,
                    'sharpe': sharpe,
                    'var': var,
                    'mdd': mdd
                }
    
    
    weights = portfolio_values / portfolio_values.sum()
    concentration = np.sum(weights ** 2)  
    
    sigma = np.clip(0.15 + concentration * 0.2, 0.03, 0.50)
    beta = 1.0
    sharpe = np.clip(0.8 - concentration * 0.3, 0.0, 4.0)
    var = np.clip(0.05 + concentration * 0.1, 0.005, 0.40)
    mdd = np.clip(0.10 + concentration * 0.15, 0.01, 0.60)
    
    return {'sigma': sigma, 'beta': beta, 'sharpe': sharpe, 'var': var, 'mdd': mdd}

def calculate_target_v_from_questions(c1, c2, c3, t1, t2, t3):

    capacity_score = (c1 + c2 + c3) / 9.0  
    tolerance_score = (t1 + t2 + t3) / 9.0  
    
    W_CAPACITY = 0.60
    W_TOLERANCE = 0.40
    
    V = (W_CAPACITY * capacity_score + W_TOLERANCE * tolerance_score) * 100
    return V
    #     question_weights = {1:[10,6,2], 2: [10,5,1], 3: [8,4,2]}

    #     raw_capacity_score = question_weights[1][c1-1] +question_weights[2][c2-1] + question_weights[3][c3-1]
    #     raw_tolerance_score = question_weights[1][t1-1]+ question_weights[2][t2-1]+ question_weights[3][t3-1]
    #     normalized_capacity = (raw_capacity_score / 28) * 100 
    #     normalized_tolerance = (raw_tolerance_score / 28) * 100 
        
    #     # capacity_score = (c1 + c2 + c3) / 9.0  # Normalize to 0-1
    #     # tolerance_score = (t1 + t2 + t3) / 9.0  # Normalize to 0-1
        
    #     W_CAPACITY = 0.60
    #     W_TOLERANCE = 0.40
    #     V = (W_CAPACITY * normalized_capacity) + (W_TOLERANCE * normalized_tolerance) 

    # # V = (W_CAPACITY * capacity_score + W_TOLERANCE * tolerance_score) * 100
    #     return V

#class PortfolioEnv(gym.Env):
    def __init__(self, initial_portfolio, target_v):
        super().__init__()
        self.assets = ['SPY','AAPL', 'GOOGL', 'MSFT', 'META', 'NVDA']
        self.n_assets = 6
        self.target_v = target_v
        self.initial_portfolio = np.array(initial_portfolio, dtype=np.float32)
        self.portfolio = self.initial_portfolio.copy()
        
        self.max_steps = 50 
        self.current_step = 0
        
        self.initial_total_value = self.initial_portfolio.sum()
        
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
        
        metrics = calculate_portfolio_metrics(self.portfolio)
        current_z = calculate_z(metrics)
        
        obs = np.concatenate([normalized_portfolio, [self.target_v / 100.0, current_z / 100.0]])
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)
        obs = np.clip(obs, -10, 10)
        return obs.astype(np.float32)

    def step(self, action):
        # **CHANGE 1: Adjusted Action Scaling**
        # Scale action to a percentage of the initial total portfolio value.
        # Max movement is 10% of total portfolio value in one step (1000/15000 approx 6.6%)
        # Let's keep it based on the initial scale but make the agent's maximum action smaller.
        # Max action dollar amount: 10% of initial total value (e.g., $1500 for $15k portfolio)
        MAX_MOVE_DOLLAR = self.initial_total_value * 0.10 
        
        # The agent output is -1 to 1. We scale it by a smaller max move.
        scaled_action = action * MAX_MOVE_DOLLAR / self.max_steps 
        # The max move in *one step* will be 1/50 of the total allowed max move. 
        # This forces the agent to take multiple, smaller steps.
        
        # Store the magnitude of the action before applying constraints
        action_magnitude = np.sum(np.abs(scaled_action))

        # Apply action with constraint: can't sell more than you have
        for i in range(self.n_assets):
            if scaled_action[i] < 0:  # Selling
                max_sell = self.portfolio[i]
                scaled_action[i] = max(scaled_action[i], -max_sell)
        
        # Apply action
        new_portfolio = self.portfolio + scaled_action
        new_portfolio = np.maximum(new_portfolio, 0)
        
        # Constraint: Total portfolio value cannot exceed initial total value
        initial_total = self.initial_total_value
        new_total = new_portfolio.sum()
        
        if new_total > initial_total:
            # Scale down to maintain total value
            scale_factor = initial_total / new_total
            new_portfolio = new_portfolio * scale_factor
        
        # **CHANGE 2: Add a Soft Constraint/Penalty for Extreme Actions**
        # The true portfolio change is new_portfolio - self.portfolio.
        # We can penalize any single asset being completely zeroed out if it wasn't zero initially.
        sell_out_penalty = 0
        for i in range(self.n_assets):
            if self.initial_portfolio[i] > 10 and new_portfolio[i] < 1: # If initially > $10 and now < $1
                sell_out_penalty += 10 # Harsh penalty

        self.portfolio = new_portfolio

        # Calculate Z
        metrics = calculate_portfolio_metrics(self.portfolio)
        portfolio_z = calculate_z(metrics)

        # **CHANGE 3: Improved Reward Function**
        diff = abs(self.target_v - portfolio_z)
        # 1. Base Reward: Negative of the difference (closer to zero is better)
        reward_z = -diff / 100.0
        
        # 2. **Action Cost/Penalty (for realistic rebalancing):** # Penalize large total moves to encourage small, efficient steps.
        # action_magnitude is the sum of |buys| + |sells| for this step.
        ACTION_COST_FACTOR = 0.0001
        reward_action_cost = -action_magnitude * ACTION_COST_FACTOR 
        
        # 3. Final Reward
        reward = reward_z + reward_action_cost - sell_out_penalty
        
        self.current_step += 1
        done = self.current_step >= self.max_steps

        return self._get_observation(), reward, done, {}
class PortfolioEnv(gym.Env):
    def __init__(self, initial_portfolio, target_v):
        super().__init__()
        self.assets = ['SPY','AAPL', 'GOOGL', 'MSFT', 'META', 'NVDA']
        self.n_assets = 6
        self.target_v = target_v
        self.initial_portfolio = np.array(initial_portfolio, dtype=np.float32)
        self.portfolio = self.initial_portfolio.copy()
        
        self.max_steps = 50 
        self.current_step = 0
        
        # Store initial total value for constraints and action scaling base
        self.initial_total_value = self.initial_portfolio.sum()
        
        # Action: change in dollar amount for each asset - Kept range -1.0 to 1.0
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)
        
        # Observation: [normalized portfolio values (5), target_v (1), current_z (1)]
        self.observation_space = spaces.Box(low=-10, high=10, shape=(self.n_assets + 2,), dtype=np.float32)

    def reset(self):
        self.portfolio = self.initial_portfolio.copy()
        self.current_step = 0
        return self._get_observation()

    def _get_observation(self):
        total = self.portfolio.sum() + 1e-9
        normalized_portfolio = self.portfolio / total
        normalized_portfolio = np.nan_to_num(normalized_portfolio, nan=0.0, posinf=1.0, neginf=0.0)
        
        metrics = calculate_portfolio_metrics(self.portfolio)
        current_z = calculate_z(metrics)
        
        obs = np.concatenate([normalized_portfolio, [self.target_v / 100.0, current_z / 100.0]])
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)
        obs = np.clip(obs, -10, 10)
        return obs.astype(np.float32)

    def step(self, action):
        # **CHANGE 1: Adjusted Action Scaling**
        # Scale action to a percentage of the initial total portfolio value.
        # Max movement is 10% of total portfolio value in one step (1000/15000 approx 6.6%)
        # Let's keep it based on the initial scale but make the agent's maximum action smaller.
        # Max action dollar amount: 10% of initial total value (e.g., $1500 for $15k portfolio)
        MAX_MOVE_DOLLAR = self.initial_total_value * 0.10 
        
        # The agent output is -1 to 1. We scale it by a smaller max move.
        scaled_action = action * MAX_MOVE_DOLLAR / self.max_steps 
        # The max move in *one step* will be 1/50 of the total allowed max move. 
        # This forces the agent to take multiple, smaller steps.
        
        # Store the magnitude of the action before applying constraints
        action_magnitude = np.sum(np.abs(scaled_action))

        # Apply action with constraint: can't sell more than you have
        for i in range(self.n_assets):
            if scaled_action[i] < 0:  # Selling
                max_sell = self.portfolio[i]
                scaled_action[i] = max(scaled_action[i], -max_sell)
        
        # Apply action
        new_portfolio = self.portfolio + scaled_action
        new_portfolio = np.maximum(new_portfolio, 0)
        
        # Constraint: Total portfolio value cannot exceed initial total value
        initial_total = self.initial_total_value
        new_total = new_portfolio.sum()
        
        if new_total > initial_total:
            # Scale down to maintain total value
            scale_factor = initial_total / new_total
            new_portfolio = new_portfolio * scale_factor
        
        # **CHANGE 2: Add a Soft Constraint/Penalty for Extreme Actions**
        # The true portfolio change is new_portfolio - self.portfolio.
        # We can penalize any single asset being completely zeroed out if it wasn't zero initially.
        sell_out_penalty = 0
        for i in range(self.n_assets):
            if self.initial_portfolio[i] > 10 and new_portfolio[i] < 1: # If initially > $10 and now < $1
                sell_out_penalty += 10 # Harsh penalty

        self.portfolio = new_portfolio

        # Calculate Z
        metrics = calculate_portfolio_metrics(self.portfolio)
        portfolio_z = calculate_z(metrics)

        # **CHANGE 3: Improved Reward Function**
        diff = abs(self.target_v - portfolio_z)
        # 1. Base Reward: Negative of the difference (closer to zero is better)
        reward_z = -diff / 100.0
        
        # 2. **Action Cost/Penalty (for realistic rebalancing):** # Penalize large total moves to encourage small, efficient steps.
        # action_magnitude is the sum of |buys| + |sells| for this step.
        ACTION_COST_FACTOR = 0.0001
        reward_action_cost = -action_magnitude * ACTION_COST_FACTOR 
        
        # 3. Final Reward
        reward = reward_z + reward_action_cost - sell_out_penalty
        
        self.current_step += 1
        done = self.current_step >= self.max_steps

        return self._get_observation(), reward, done, {}


def optimize_portfolio(model, questions, current_portfolio):
    
    target_v = calculate_target_v_from_questions(
        questions['C1'], questions['C2'], questions['C3'],
        questions['T1'], questions['T2'], questions['T3']
    )
    
    portfolio_array = np.array([
        current_portfolio['SPY'],
        current_portfolio['AAPL'],
        current_portfolio['GOOGL'],
        current_portfolio['MSFT'],
        current_portfolio['META'],
        current_portfolio['NVDA']
    ])
    
    env = PortfolioEnv(portfolio_array, target_v)
    obs = env.reset()
    
    initial_metrics = calculate_portfolio_metrics(portfolio_array)
    initial_z = calculate_z(initial_metrics)
    print()
    print(f"Target Risk Score (V):{target_v}/100")
    print(f"Current Portfolio Risk (Z):{initial_z}/100")
    total_actions = np.zeros(6) 
    
    current_state_portfolio = portfolio_array.copy()
    
    for step in range(env.max_steps): 
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        
        step_change = env.portfolio - current_state_portfolio
        total_actions += step_change
        current_state_portfolio = env.portfolio.copy()
        
        if (step + 1) % 10 == 0:
            temp_metrics = calculate_portfolio_metrics(env.portfolio)
            temp_z = calculate_z(temp_metrics)
            temp_diff = abs(target_v - temp_z)
            # print(f"  Step {step + 1}: Z = {temp_z:.2f}, Diff = {temp_diff:.2f}, Reward = {reward:.4f}")
        
        if done:
            break
    
    final_portfolio = env.portfolio
    return final_portfolio,portfolio_array
    

if __name__ == "__main__":
    
    load_market_data()
    
    model = PPO.load("final_model")
    
    user_questions = {
        'C1': 3, 'C2': 2, 'C3': 2, 
        'T1': 3, 'T2': 3, 'T3': 2 
    }
    user_portfolio = {
        'SPY' : 2000, 'AAPL': 5000, 'GOOGL': 3000, 
        'MSFT': 2000, 'META': 1000, 'NVDA': 4000
    }
    
    final_portfolio, portfolio_array= optimize_portfolio(model, user_questions, user_portfolio) # --> this is the final return of the method, display the stuff below in the ui
    total_changes = final_portfolio - portfolio_array
    
    final_metrics = calculate_portfolio_metrics(final_portfolio)
    final_z = calculate_z(final_metrics)

    print(f"Optimized Portfolio Risk (Z):{final_z:.2f}/100")
    print()
    assets = ['SPY','AAPL', 'GOOGL', 'MSFT', 'META', 'NVDA']
    recommendations = {}
    
    
    total_buys = 0
    total_sells = 0
    
    for i, asset in enumerate(assets):
        change = total_changes[i]
        new_value = final_portfolio[i] 
        recommendations[asset] = {
            'current': portfolio_array[i],
            'change': change,
            'new': new_value
        }
        
        if change > 0:
            total_buys += change
        else:
            total_sells += abs(change)
        
        if abs(change) > 10: 
            action_str = "BUY" if change > 0 else "SELL"
            pct_change = (change / portfolio_array[i] * 100) if portfolio_array[i] > 0 else (100 if change > 0 else 0)
            
            print(f"{asset:6s}: {action_str:4s} ${abs(change):8,.2f} ({pct_change:+6.1f}%)")
            print(f"         Current: ${portfolio_array[i]:8,.2f} → New: ${new_value:8,.2f}")
        else:
            print(f"{asset:6s}: HOLD (no significant change)")
    
    initial_total = sum(portfolio_array)
    new_total = sum(final_portfolio)

    