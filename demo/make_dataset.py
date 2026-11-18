

from pathlib import Path

import numpy as np

N, DRIFT_AT, NEW_REGION_AT, SEED = 12_000, 7_000, 4_000, 2026


def make(n=N, seed=SEED):
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 71, n).astype(float)
    income = np.round(rng.lognormal(np.log(65), 0.45, n), 1)          
    loan = np.round(rng.lognormal(np.log(25), 0.6, n), 1)             
    credit = np.clip(np.round(rng.normal(670, 70, n)), 300, 850)
    dti = np.round(np.clip(rng.beta(2.2, 5, n) * 1.1, 0, 1), 3)       
    late = rng.poisson(0.7, n).astype(float)
    purpose = rng.choice(["home", "car", "education", "business"], n, p=[.35, .3, .15, .2])
    region = rng.choice(["north", "south", "east", "west"], n)
    late_idx = np.arange(n) >= NEW_REGION_AT
    online = late_idx & (rng.random(n) < 0.25)
    region = np.where(online, "online", region)

    t = np.arange(n)
    before = (-0.018 * (credit - 670) + 4.0 * (dti - 0.32) + 0.9 * late
              + 0.9 * (purpose == "business") - 1.2 * np.log(income / 65)
              + 0.6 * (loan / income > 0.6) - 1.5)
    after = (7.0 * (dti - 0.25) + 1.2 * late - 0.005 * (credit - 670)
             + 1.6 * (region == "online") + 1.0 * (age < 30) - 0.5 * (years > 5) - 1.7)
    risk = np.where(t < DRIFT_AT, before, after)
    p = 1 / (1 + np.exp(-2.2 * risk))
    default = (rng.random(n) < p).astype(int)

    cols = {"age": age, "income": income, "loan_amount": loan, "credit_score": credit,
            "debt_to_income": dti, "years_employed": years, "late_payments": late}
    for name in ("income", "credit_score", "years_employed", "debt_to_income"):
        miss = rng.random(n) < 0.04
        cols[name] = np.where(miss, np.nan, cols[name])
    region = np.where(rng.random(n) < 0.02, "", region)
    return cols, region, purpose, default


def write(path):
    cols, region, purpose, default = make()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(cols) + ["region", "loan_purpose", "default"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(",".join(names) + "\n")
        for i in range(len(default)):
            num = ["" if np.isnan(cols[c][i]) else f"{cols[c][i]:g}" for c in cols]
            fh.write(",".join(num + [region[i], purpose[i], str(default[i])]) + "\n")
    return path


if __name__ == "__main__":
    out = write(Path(__file__).parent / "data" / "loan_stream.csv")
    print(f"wrote {out}")
