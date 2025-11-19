
import argparse
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import pandas as pd


@dataclass
class PolicyStats:
    name: str
    total_steps: int = 0               # total rows processed
    matched_steps: int = 0             # times chosen arm == logged arm
    matched_clicks: int = 0            # clicks on matched rows
    ctr_history: List[Tuple[int, float]] = field(default_factory=list)


class EpsilonGreedy: # epsilon-greedy bandit policy
    def __init__(self, n_arms: int, epsilon: float = 0.1, name: str = "epsilon_greedy"):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.name = name
        self.counts = [0] * n_arms
        self.sums = [0.0] * n_arms
        self.stats = PolicyStats(name=name)

    def select_arm(self) -> int:
        # if there are arms never tried, pick them first to initialize
        for i in range(self.n_arms):
            if self.counts[i] == 0:
                return i
        if random.random() < self.epsilon:
            return random.randrange(self.n_arms)
        # exploit: choose arm with highest empirical mean
        means = [self.sums[i] / self.counts[i] for i in range(self.n_arms)]
        return int(max(range(self.n_arms), key=lambda i: means[i]))

    def update(self, chosen_arm: int, reward: float, matched: bool):
        self.stats.total_steps += 1
        # Only update bandit estimates when we actually observe a reward
        if matched:
            self.counts[chosen_arm] += 1
            self.sums[chosen_arm] += reward
            self.stats.matched_steps += 1
            if reward > 0:
                self.stats.matched_clicks += 1

        # track CTR over matched data
        if self.stats.matched_steps > 0 and self.stats.total_steps % 1000 == 0:
            ctr = self.stats.matched_clicks / self.stats.matched_steps
            self.stats.ctr_history.append((self.stats.total_steps, ctr))


class UCB1: # UCB1 bandit policy
    def __init__(self, n_arms: int, name: str = "ucb1"):
        self.n_arms = n_arms
        self.name = name
        self.counts = [0] * n_arms
        self.sums = [0.0] * n_arms
        self.total_count = 0
        self.stats = PolicyStats(name=name)

    def select_arm(self) -> int:
        # play each arm at least once
        for i in range(self.n_arms):
            if self.counts[i] == 0:
                return i
        # compute UCB value for each arm
        ucb_values = []
        for i in range(self.n_arms):
            mean = self.sums[i] / self.counts[i]
            bonus = math.sqrt(2.0 * math.log(self.total_count) / self.counts[i])
            ucb_values.append(mean + bonus)
        return int(max(range(self.n_arms), key=lambda i: ucb_values[i]))

    def update(self, chosen_arm: int, reward: float, matched: bool):
        self.stats.total_steps += 1
        self.total_count += 1
        if matched:
            self.counts[chosen_arm] += 1
            self.sums[chosen_arm] += reward
            self.stats.matched_steps += 1
            if reward > 0:
                self.stats.matched_clicks += 1

        if self.stats.matched_steps > 0 and self.stats.total_steps % 1000 == 0:
            ctr = self.stats.matched_clicks / self.stats.matched_steps
            self.stats.ctr_history.append((self.stats.total_steps, ctr))


class ThompsonSampling: # Thompson sampling bandit policy
    def __init__(self, n_arms: int, name: str = "thompson_sampling"):
        self.n_arms = n_arms
        self.name = name
        # Beta prior parameters for each arm (alpha, beta)
        self.alpha = [1.0] * n_arms
        self.beta = [1.0] * n_arms
        self.stats = PolicyStats(name=name)

    def select_arm(self) -> int:
        samples = [random.betavariate(self.alpha[i], self.beta[i]) for i in range(self.n_arms)]
        return int(max(range(self.n_arms), key=lambda i: samples[i]))

    def update(self, chosen_arm: int, reward: float, matched: bool):
        self.stats.total_steps += 1
        if matched:
            # reward is 0/1 click
            self.alpha[chosen_arm] += reward
            self.beta[chosen_arm] += 1.0 - reward
            self.stats.matched_steps += 1
            if reward > 0:
                self.stats.matched_clicks += 1

        if self.stats.matched_steps > 0 and self.stats.total_steps % 1000 == 0:
            ctr = self.stats.matched_clicks / self.stats.matched_steps
            self.stats.ctr_history.append((self.stats.total_steps, ctr))


class SlidingWindowThompsonSampling: # Sliding window Thompson sampling bandit policy

    def __init__(self, n_arms: int, window_size: int = 5000, name: str = "sw_ts"):
        self.n_arms = n_arms
        self.window_size = window_size
        self.name = name
        # maintain counts and sums over window
        self.counts = [0] * n_arms
        self.sums = [0.0] * n_arms
        self.buffer: List[Tuple[int, float]] = []  # list of (arm, reward)
        self.stats = PolicyStats(name=name)

    def select_arm(self) -> int:
        alpha = []
        beta = []
        for i in range(self.n_arms):
            successes = self.sums[i]
            trials = self.counts[i]
            alpha.append(1.0 + successes)
            beta.append(1.0 + max(trials - successes, 0.0))
        samples = [random.betavariate(alpha[i], beta[i]) for i in range(self.n_arms)]
        return int(max(range(self.n_arms), key=lambda i: samples[i]))

    def update(self, chosen_arm: int, reward: float, matched: bool):
        self.stats.total_steps += 1
        if matched:
            # add new observation to window
            self.buffer.append((chosen_arm, reward))
            self.counts[chosen_arm] += 1
            self.sums[chosen_arm] += reward
            self.stats.matched_steps += 1
            if reward > 0:
                self.stats.matched_clicks += 1

            # drop oldest if window too large
            if len(self.buffer) > self.window_size:
                old_arm, old_reward = self.buffer.pop(0)
                self.counts[old_arm] -= 1
                self.sums[old_arm] -= old_reward

        if self.stats.matched_steps > 0 and self.stats.total_steps % 1000 == 0:
            ctr = self.stats.matched_clicks / self.stats.matched_steps
            self.stats.ctr_history.append((self.stats.total_steps, ctr))


def run_offline_replay(
    df: pd.DataFrame,
    arm_col: str = "arm_id",
    reward_col: str = "click",
    seed: int = 42,
    window_size: int = 5000,
) -> Dict[str, PolicyStats]: # run offline bandit evaluation on logged Avazu subset
    # set random seed
    random.seed(seed)

    # map arm ids to indices 0..K-1
    arm_ids = sorted(df[arm_col].astype(str).unique().tolist())
    arm_to_idx = {a: i for i, a in enumerate(arm_ids)}
    n_arms = len(arm_ids)

    print(f"Found {n_arms} arms: {arm_ids}")

    # instantiate policies
    eps_policy = EpsilonGreedy(n_arms=n_arms, epsilon=0.1, name="epsilon_greedy_0.1")
    ucb_policy = UCB1(n_arms=n_arms, name="ucb1")
    ts_policy = ThompsonSampling(n_arms=n_arms, name="thompson_sampling")
    sw_ts_policy = SlidingWindowThompsonSampling(
        n_arms=n_arms, window_size=window_size, name=f"sw_ts_{window_size}"
    )

    policies = [eps_policy, ucb_policy, ts_policy, sw_ts_policy]

    # ensure chronological order
    if "hour" in df.columns:
        df = df.sort_values("hour").reset_index(drop=True)

    for idx, row in df.iterrows():
        logged_arm = arm_to_idx[str(row[arm_col])]
        reward = float(row[reward_col])

        for policy in policies:
            chosen = policy.select_arm()
            matched = chosen == logged_arm
            policy.update(chosen, reward, matched)

        if (idx + 1) % 5000 == 0:
            print(f"Processed {idx + 1} rows...")

    results = {p.name: p.stats for p in policies}
    return results


def save_results(results: Dict[str, PolicyStats], output_path: str): #
    rows = []
    for name, stats in results.items():
        if stats.matched_steps > 0:
            ctr = stats.matched_clicks / stats.matched_steps
        else:
            ctr = float("nan")
        rows.append(
            {
                "policy": name,
                "total_steps": stats.total_steps,
                "matched_steps": stats.matched_steps,
                "matched_fraction": stats.matched_steps / stats.total_steps
                if stats.total_steps > 0
                else float("nan"),
                "matched_clicks": stats.matched_clicks,
                "ctr_on_matched": ctr,
            }
        )
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_path, index=False)
    print("\n=== Summary ===")
    print(summary_df)
    print(f"\nSaved summary to: {output_path}")


def maybe_plot(results: Dict[str, PolicyStats], output_png: str): # plot the CTR history
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("Could not import matplotlib, skipping plots.", e)
        return

    plt.figure(figsize=(8, 5))
    for name, stats in results.items():
        if not stats.ctr_history:
            continue
        xs = [t for (t, _) in stats.ctr_history]
        ys = [ctr for (_, ctr) in stats.ctr_history]
        plt.plot(xs, ys, label=name)

    plt.xlabel("Total steps processed (rows)")
    plt.ylabel("CTR on matched impressions")
    plt.title("Cumulative CTR vs steps (offline replay)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    print(f"Saved CTR history plot to: {output_png}")


def main():
    parser = argparse.ArgumentParser(
        description="Run epsilon-greedy, UCB1, TS, and Sliding-Window TS on avazu_bandit_subset.csv"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to avazu_bandit_subset.csv",
    )
    parser.add_argument(
        "--summary_path",
        type=str,
        default="bandit_policy_summary.csv",
        help="Where to save the summary CSV (default: bandit_policy_summary.csv)",
    )
    parser.add_argument(
        "--plot_path",
        type=str,
        default="bandit_ctr_history.png",
        help="Where to save the CTR history plot (default: bandit_ctr_history.png)",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=200000,
        help="Max number of rows to use from the subset (default: 200000)",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=5000,
        help="Sliding window size for SW-TS (default: 5000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    print(f"Loading data from {args.data_path} (max_rows={args.max_rows})...")
    df = pd.read_csv(args.data_path, nrows=args.max_rows)

    results = run_offline_replay(
        df,
        arm_col="arm_id",
        reward_col="click",
        seed=args.seed,
        window_size=args.window_size,
    )

    save_results(results, args.summary_path)
    maybe_plot(results, args.plot_path)


if __name__ == "__main__":
    main()
