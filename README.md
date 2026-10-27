# Problem-Set-2
***Model: Claude ~ Opus 5.5 ; Date: 10/26**

***Version 1 prompt:*** 
[Prep] Act as a survey statistician who works on non-response
adjustment for product research.

[Request] I have a simulated user population with satisfaction scores
and two covariates, engagement_minutes and had_support_ticket, plus a
non-response mechanism where the probability of responding increases
with satisfaction. Write Python that implements inverse-probability
weighting to correct the observed survey mean, using a logistic model
of response on the covariates.

[Iterate] numpy, pandas, scikit-learn or statsmodels. Reuse my
variable names. Return the weighted estimate and its bootstrap
interval.

[Mechanism Check] Explain precisely which assumption makes this
correction valid, and what happens to it when response depends on
satisfaction ITSELF rather than only on the covariates.

[Evaluate] Given my mechanism, will this correction recover the true
mean? Answer yes or no and justify it.

***Reply:***
The script uses your variable names (satisfaction, engagement_minutes, had_support_ticket). The simulation block at the top is a placeholder, so swap in your own df. The rest assumes a responded column and that the covariates are known for everyone in the sample, including people who didn’t respond. That’s what lets you fit the response model.

How it works. The script fits a logistic regression of responded on the two covariates. That gives each user an estimated response probability p̂. Each respondent’s satisfaction is weighted by 1/p̂, and the weighted mean is computed. The bootstrap resamples the whole sample, both respondents and non-respondents, and refits the logistic model on every draw. That way the interval also reflects the uncertainty from estimating the weights.

When I ran it on the placeholder simulation:

	Value
True mean	5.658
Naive survey mean	6.538
IPW estimate	6.373
95% bootstrap CI	[6.342, 6.408]

The weighting removes some of the bias, but the true mean lies well outside the interval.

Which assumption makes this valid. The correction works only if, once you know a user’s engagement_minutes and had_support_ticket, their chance of responding no longer depends on their satisfaction. Put another way, among users with the same covariate values, respondents and non-respondents have the same satisfaction distribution on average. In that case each respondent can stand in for the similar non-respondents, and the 1/p̂ weight scales them up by exactly the right amount. Two smaller conditions also matter. Every covariate profile needs a nonzero chance of responding, and the logistic model has to describe the response probabilities reasonably well.

When response depends on satisfaction itself. The weights can only adjust for what the covariates explain. In your setup, satisfaction includes a part the covariates don’t capture (the noise term). Response depends on that part too. So within any covariate group, the respondents are still the more satisfied users, and reweighting them can’t fix that. The logistic model on covariates picks up only the share of the response mechanism that runs through engagement and support tickets. The rest of the bias remains. Collecting more data won’t help, because the estimate converges to the wrong value. The bootstrap interval then gets narrow around that wrong value, as the table shows. The interval reflects sampling uncertainty, not this bias.

Will it recover the true mean under your mechanism? No. Your response probability rises with satisfaction directly, and satisfaction isn’t fully determined by the two covariates. That breaks the assumption above. The correction will move the estimate toward the truth, to the extent that the covariates are correlated with satisfaction, but some upward bias will remain. The one exception would be if satisfaction were an exact function of engagement_minutes and had_support_ticket. Then conditioning on the covariates would be the same as conditioning on satisfaction. That isn’t the situation you described.

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(3916)

# ------------------------------------------------------------------
# 1. Simulated population (replace this block with your own df)
#    Needs columns: satisfaction, engagement_minutes,
#    had_support_ticket, responded (1 = answered the survey)
# ------------------------------------------------------------------
N = 20_000
engagement_minutes = rng.gamma(shape=2.0, scale=15.0, size=N)
had_support_ticket = rng.binomial(1, 0.25, size=N)
satisfaction = (
    5.0
    + 0.03 * engagement_minutes
    - 1.0 * had_support_ticket
    + rng.normal(0, 1.5, size=N)          # part of satisfaction NOT explained by covariates
)

# Non-response: probability of responding increases with satisfaction
p_respond = 1 / (1 + np.exp(-(-4.0 + 0.6 * satisfaction)))
responded = rng.binomial(1, p_respond)

df = pd.DataFrame({
    "satisfaction": satisfaction,
    "engagement_minutes": engagement_minutes,
    "had_support_ticket": had_support_ticket,
    "responded": responded,
})
# Satisfaction is only observed for respondents
df.loc[df["responded"] == 0, "satisfaction_obs"] = np.nan
df.loc[df["responded"] == 1, "satisfaction_obs"] = df["satisfaction"]

covariates = ["engagement_minutes", "had_support_ticket"]

# ------------------------------------------------------------------
# 2. IPW estimator
# ------------------------------------------------------------------
def ipw_mean(data):
    """Fit logistic model of response on covariates, weight respondents by 1/p."""
    model = LogisticRegression(C=1e6, max_iter=1000)   # large C ≈ no penalty
    model.fit(data[covariates], data["responded"])
    p_hat = model.predict_proba(data[covariates])[:, 1]

    resp = data["responded"].to_numpy() == 1
    w = 1 / p_hat[resp]
    y = data.loc[resp, "satisfaction_obs"].to_numpy()
    return np.sum(w * y) / np.sum(w)

# ------------------------------------------------------------------
# 3. Bootstrap interval (resample the whole sample, refit the model each time)
# ------------------------------------------------------------------
def bootstrap_ci(data, B=1000, alpha=0.05):
    n = len(data)
    est = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        est[b] = ipw_mean(data.iloc[idx])
    return np.quantile(est, [alpha / 2, 1 - alpha / 2])

true_mean = df["satisfaction"].mean()
naive_mean = df["satisfaction_obs"].mean()          # respondents only
ipw_est = ipw_mean(df)
lo, hi = bootstrap_ci(df, B=500)

print(f"Response rate:       {df['responded'].mean():.3f}")
print(f"True mean:           {true_mean:.3f}")
print(f"Naive survey mean:   {naive_mean:.3f}")
print(f"IPW estimate:        {ipw_est:.3f}")
print(f"95% bootstrap CI:    [{lo:.3f}, {hi:.3f}]")
print(f"True mean inside CI? {lo <= true_mean <= hi}")


***Version 2 prompt:***
[Prep] Act as a survey statistician who works on non-response
adjustment for product research.

[Data] Do NOT simulate a new population. My data already exists in
my notebook, so write code that uses these objects directly:
- pop_before and pop_after: pandas DataFrames, 50,000 users each, with
  columns engagement_minutes, engagement_z, had_support_ticket,
  s (latent satisfaction) and satisfaction (the reported 1–5 score).
- resp_before and resp_after: boolean pandas Series, aligned to each
  DataFrame's index, True if the user answered the survey.
- rng: a numpy Generator created with np.random.default_rng(42).
  Use it for all randomness and do not create a new one or re-seed it.
- My known true population means of satisfaction are 3.290 (before)
  and 3.386 (after). The naive respondent means are 3.600 and 4.304.

[Request] My non-response mechanism makes the probability of responding
increase with satisfaction. Write Python that implements
inverse-probability weighting to correct the observed survey mean,
using a logistic model of response on the covariates
engagement_z and had_support_ticket, fitted on all 50,000 users
(respondents and non-respondents). Do this separately for the
before and after periods.

[Iterate] numpy, pandas, scikit-learn or statsmodels. Reuse my
variable names. Return the weighted estimate and its bootstrap
interval for each period (refit the logistic model inside every
bootstrap resample, B = 500). Print, for each period: the true mean,
the naive respondent mean, the weighted estimate, the 95% interval,
and whether the interval contains the true mean.

[Mechanism Check] Explain precisely which assumption makes this
correction valid, and what happens to it when response depends on
satisfaction ITSELF rather than only on the covariates.

[Evaluate] Given my mechanism, will this correction recover the true
mean? Answer yes or no and justify it.


***Reply:***
The script ipw_before_after.py works directly on pop_before, pop_after, resp_before, resp_after and rng. It doesn’t simulate anything and doesn’t create or re-seed a generator. I don’t have your data, so I checked that it runs on throwaway stand-in objects with the same structure. The numbers it prints in your notebook will be your own.

What the code does, for each period

It fits a logistic regression of resp on engagement_z and had_support_ticket using all 50,000 users, both respondents and non-respondents.
It gives each respondent a weight of 1/p̂, using their predicted response probability.
It takes the weighted mean of satisfaction over respondents only. The script never reads non-respondents’ scores, even though they’re in the DataFrame.
For the bootstrap, it draws 500 resamples of all 50,000 users with rng.integers. In each resample it refits the logistic model and recomputes the weighted mean. The 95% interval is the 2.5th to 97.5th percentile of those 500 estimates.
It prints the true mean, the naive respondent mean, the weighted estimate, the interval, and whether the interval contains the true mean. It also collects these in results_df.

The before period runs first, so the after period’s bootstrap continues from wherever rng stands at that point. If you rerun the cell without re-running the cell that creates rng, you’ll get slightly different intervals.

Which assumption makes the correction valid

The correction is valid only if response is unrelated to satisfaction once you know engagement_z and had_support_ticket. In other words, among users with the same covariate values, respondents and non-respondents must have the same satisfaction on average. Then each respondent can represent the similar users who didn’t answer, and 1/p̂ scales them up by the right amount. Two further conditions are needed. Every covariate profile must have some chance of responding, and the logistic model must describe the response probabilities reasonably well.

What happens when response depends on satisfaction itself

The weights can only correct the part of the selection that runs through the two covariates. Your latent satisfaction s presumably has a component the covariates don’t explain. If response depends on s, then within every covariate group the respondents are still the more satisfied users. The logistic model can’t see that within-group selection, so reweighting leaves it untouched.

The estimate moves from the naive mean toward the truth only as far as the covariates are related to satisfaction. The remaining bias doesn’t shrink with more users. The bootstrap interval measures sampling uncertainty around this biased estimate, so with 50,000 users it will be narrow and can easily exclude the true mean. A narrow interval here isn’t a sign that the bias has been fixed.

Will it recover the true mean under your mechanism? No.

Your response probability increases with satisfaction directly. Unless s were an exact function of engagement_z and had_support_ticket, conditioning on the covariates doesn’t remove that dependence, and the assumption above fails. Expect each weighted estimate to land between the naive mean (3.600 before, 4.304 after) and the true mean (3.290, 3.386). Expect the intervals to most likely miss the truth.

The after period should show the larger leftover gap. Its naive bias is much bigger (+0.918 versus +0.310), which suggests response depends more strongly on satisfaction in that period. The covariates can only absorb part of that dependence.

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Uses objects already in your notebook:
#   pop_before, pop_after, resp_before, resp_after, rng
# No new data is simulated and rng is not re-created or re-seeded.

covariates = ["engagement_z", "had_support_ticket"]

true_means = {"before": 3.290, "after": 3.386}


def ipw_mean(pop, resp):
    """Fit logistic model of response on covariates using ALL users,
    then weight respondents' satisfaction by 1 / p_hat."""
    X = pop[covariates].to_numpy()
    r = resp.to_numpy().astype(int)

    model = LogisticRegression(C=1e6, max_iter=1000)   # large C ~ no penalty
    model.fit(X, r)
    p_hat = model.predict_proba(X)[:, 1]

    is_resp = r == 1
    w = 1.0 / p_hat[is_resp]
    y = pop["satisfaction"].to_numpy()[is_resp]         # only respondents' scores are used
    return np.sum(w * y) / np.sum(w)


def bootstrap_ci(pop, resp, B=500, alpha=0.05):
    """Resample all users (respondents and non-respondents) with replacement,
    refit the logistic model in every resample."""
    n = len(pop)
    est = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)                # positional indices
        est[b] = ipw_mean(pop.iloc[idx], resp.iloc[idx])
    lo, hi = np.quantile(est, [alpha / 2, 1 - alpha / 2])
    return lo, hi


results = {}
for period, pop, resp in [("before", pop_before, resp_before),
                          ("after",  pop_after,  resp_after)]:
    resp = resp.reindex(pop.index)                       # guarantee alignment
    naive = pop.loc[resp, "satisfaction"].mean()
    est = ipw_mean(pop, resp)
    lo, hi = bootstrap_ci(pop, resp, B=500)
    true_mean = true_means[period]

    results[period] = {
        "true_mean": true_mean,
        "naive_mean": naive,
        "ipw_estimate": est,
        "ci_low": lo,
        "ci_high": hi,
        "ci_contains_true": lo <= true_mean <= hi,
    }

    print(f"--- {period} ---")
    print(f"True mean:             {true_mean:.3f}")
    print(f"Naive respondent mean: {naive:.3f}")
    print(f"IPW estimate:          {est:.3f}")
    print(f"95% bootstrap CI:      [{lo:.3f}, {hi:.3f}]")
    print(f"CI contains true mean? {lo <= true_mean <= hi}")
    print()

results_df = pd.DataFrame(results).T
results_df
