# %%
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from datetime import date, timedelta
from calendar import isleap

@dataclass
class StoreConfig:
    shelf_size: int = 100                # capacity z
    replenishment_rate: int = 4      # max units added per day
    hist_length: int = 10          #n
    product_profit:  float = 0.5            
    holding_cost: float = 0.05         # c per unit per day
    horizon_days: int = 365              # episode length cap
    start_date: date = date(2021, 1, 1)  # per-env start
    end_date: date = date(2025, 12, 31)
    start_inventory: float = 100.0
    seed: Optional[int] = None


@dataclass
class Demand:
    base: float = 10.0
    weekend_multiplier: float = 1.3
    yearly_amplitude: float = 0.25
    phase_shift: float = np.pi / 12  # matches your function (subtract in formula)
    seed: Optional[int] = None

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def _rate(self, dt: date) -> float:
        # Weekend bump
        weekly = self.weekend_multiplier if dt.weekday() >= 5 else 1.0
        # Yearly seasonality (leap-year aware)
        doy = dt.timetuple().tm_yday
        max_doy = 366 if isleap(dt.year) else 365
        yearly = 1.0 + self.yearly_amplitude * np.sin(2 * np.pi * (doy / max_doy) - self.phase_shift)
        lam = max(1e-6, self.base * weekly * yearly)
        return lam

    def sample(self, dt: date) -> float:
        """Sample a single-day demand (Poisson)."""
        return float(self.rng.poisson(lam=self._rate(dt)))

    def series(self, start: date, days: int) -> List[float]:
        """Sample a consecutive series of daily demands."""
        out = []
        for i in range(days):
            out.append(self.sample(start + timedelta(days=i)))
        return out

    def __call__(self, dt: date) -> float:
        """Allow instance to be called like a function."""
        return self.sample(dt)

    def reseed(self, seed: Optional[int]) -> None:
        """Reset RNG for reproducibility."""
        self.seed = seed
        self.rng = np.random.default_rng(seed)


class StoreEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    
    def __init__(
            self,
            config: StoreConfig = StoreConfig(),
            demand: Demand = Demand(seed=42)
        ):
        super().__init__()
        self.cfg = config
        self.rng = np.random.default_rng(self.cfg.seed)

        n = self.cfg.hist_length

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        
        self.observation_space = spaces.Dict({
                "inventory": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(n,),         # n-step history
                    dtype=np.float32
                ),
                "doy": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(n,),
                    dtype=np.float32
                ),
                "dow": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(n,),
                    dtype=np.float32
                ),
                "moy": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(n,),
                    dtype=np.float32
                ),
            })

        self._demand = demand
        
        # Initialize state variables (will be set in reset)
        self.current_date = None
        self.current_stock = None
        self.inventory_stock = {}

    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        # Reset state
        self.current_date = self.cfg.start_date
        self.current_stock = self.cfg.start_inventory
        
        # Clear and reinitialize inventory history
        self.inventory_stock = {}
        
        # Initialize historical inventory (before start_date)
        # This represents the inventory levels from the past hist_length days
        for i in range(1, self.cfg.hist_length + 1):
            past_date = self.cfg.start_date - timedelta(days=i)
            if past_date in self.inventory_stock:
                print(f"[RESET-HISTORY] WARNING: Date {past_date} already in inventory_stock!")
            self.inventory_stock[past_date] = self.cfg.start_inventory
        
        # Simulate the first replenishment_rate days to get initial observation
        current_date = self.current_date
        next_demand = self._demand.series(current_date, days=self.cfg.replenishment_rate)

        for i in range(self.cfg.replenishment_rate):
            self.current_stock = max(0, self.current_stock - next_demand[i])
            
            if current_date in self.inventory_stock:
                print(f"[RESET-SIMULATE] WARNING: Date {current_date} already in inventory_stock!")
            self.inventory_stock[current_date] = self.current_stock
            current_date = current_date + timedelta(1)
        
        # Update current date to the last simulated day
        self.current_date = current_date - timedelta(1)

        # Build observation (most recent day first, going backwards)
        inv_hist, doy_hist, dow_hist, moy_hist = [], [], [], []
        
        obs_date = self.current_date
        for _ in range(self.cfg.hist_length):
            year = obs_date.year
            total_days = 366 if isleap(year) else 365

            doy_hist.append(obs_date.timetuple().tm_yday / total_days)
            dow_hist.append((obs_date.weekday() + 1) / 7)
            moy_hist.append(obs_date.month / 12)
            inv_hist.append(self.inventory_stock[obs_date] / self.cfg.shelf_size)
            
            obs_date -= timedelta(days=1)

        observation = {
                "inventory": np.array(inv_hist, dtype=np.float32),
                "doy":       np.array(doy_hist, dtype=np.float32),
                "dow":       np.array(dow_hist, dtype=np.float32),
                "moy":       np.array(moy_hist, dtype=np.float32),
            }

        return observation, {}

    def step(self, action: np.ndarray):
        # --- parse action robustly ---
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        assert a.size == 1, f"Action must be shape (1,), got {action!r}"
        a_frac = float(np.clip(a[0], 0.0, 1.0))

        # --- apply replenishment (one-shot per block) ---
        remaining_capacity = max(0.0, self.cfg.shelf_size - self.current_stock)
        a_units = a_frac * remaining_capacity
        self.current_stock = min(self.cfg.shelf_size, self.current_stock + a_units)

        # --- figure out how many days to simulate in this block ---
        start_day = self.current_date + timedelta(days=1)
        remaining_days = (self.cfg.end_date - self.current_date).days

        # If no days remain, terminate immediately
        if remaining_days <= 0:
            observation = {
                "inventory": np.array([self.current_stock / self.cfg.shelf_size] * self.cfg.hist_length, dtype=np.float32),
                "doy":       np.zeros(self.cfg.hist_length, dtype=np.float32),
                "dow":       np.zeros(self.cfg.hist_length, dtype=np.float32),
                "moy":       np.zeros(self.cfg.hist_length, dtype=np.float32),
            }
            info = {
                "action_fraction": a_frac,
                "action_units": a_units,
                "unmet_demand": 0.0,
                "holding_cost": 0.0,
                "last_demands": [],
                "last_day": self.current_date,
            }
            return observation, 0.0, True, False, info

        # simulate min(replenishment_rate, remaining_days)
        block_days = min(self.cfg.replenishment_rate, remaining_days)
        demands = self._demand.series(start_day, block_days)

        # --- roll days ---
        U = 0.0  # unmet demand penalty (forgone profit)
        C = 0.0  # holding cost
        day = start_day
        for d in demands:
            pre_inv = self.current_stock
            shortfall = max(0.0, d - pre_inv)
            end_inv = max(0.0, pre_inv - d)

            U += float(self.cfg.product_profit) * shortfall
            C += float(self.cfg.holding_cost) * end_inv

            self.current_stock = end_inv
            
            if day in self.inventory_stock:
                print(f"[STEP] WARNING: Date {day} already in inventory_stock!")
            self.inventory_stock[day] = end_inv
            day += timedelta(days=1)

        # last processed day
        self.current_date = day - timedelta(days=1)

        # ---------- Build observation (most recent EOD first) ----------
        inv_hist, doy_hist, dow_hist, moy_hist = [], [], [], []
        dcur = self.current_date
        for _ in range(self.cfg.hist_length):
            inv_eod = self.inventory_stock.get(dcur, self.cfg.start_inventory)
            inv_hist.append(inv_eod / self.cfg.shelf_size)

            max_doy = 366 if isleap(dcur.year) else 365
            doy_hist.append(dcur.timetuple().tm_yday / max_doy)
            dow_hist.append((dcur.weekday() + 1) / 7)
            moy_hist.append(dcur.month / 12)
            dcur -= timedelta(days=1)

        observation = {
            "inventory": np.array(inv_hist, dtype=np.float32),
            "doy":       np.array(doy_hist, dtype=np.float32),
            "dow":       np.array(dow_hist, dtype=np.float32),
            "moy":       np.array(moy_hist, dtype=np.float32),
        }

        # Reward: penalize unmet demand + holding cost
        reward = -(U + C)

        # --- termination / truncation logic ---
        # Terminated if we've exactly reached end_date
        terminated = (self.current_date >= self.cfg.end_date)

        # Truncated if we *couldn't* simulate a full block because we ran out of calendar before end_date
        truncated = (block_days < self.cfg.replenishment_rate) and not terminated

        info = {
            "action_fraction": a_frac,
            "action_units": a_units,
            "unmet_demand": U,
            "holding_cost": C,
            "last_demands": demands,
            "last_day": self.current_date,
        }
        return observation, float(reward), bool(terminated), bool(truncated), info


gym.register(id="StoreEnv-v0", entry_point=StoreEnv)


if __name__ == "__main__":
    config = StoreConfig()
    demand = Demand(seed=42)
    env = StoreEnv(config, demand)
    
# %%