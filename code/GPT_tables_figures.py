# imports
import pandas as pd
import numpy as np
import plotly.express as px
import os
import plotly.io as pio
import statsmodels.api as sm


pio.templates.default = "plotly_white"

"""
This Python script generates the tables and figures for the GPT paper.
"""


# Read in Results
df = pd.read_csv(os.path.join("..", "data", "gpt_dictator_results.csv"))
# Keep just the first 500 observations
# note that due to model interruptions, 501 obs total
df = df[:500]

# Find alllocation in the model's response
df["Allocation"] = df["model_answer"].str.extract("(\d+)").astype(float)
df["Allocation"].replace({50: 5}, inplace=True)  # this cases it's saying split 50-50

# Figure 1: The distribution of allocations in the GPT experiment
fig = px.histogram(
    df, x="Allocation", nbins=10, range_x=[0, 10], histnorm="probability"
)
fig.update_layout(
    xaxis=dict(tickmode="linear", tick0=0, dtick=1),
    font=dict(family="Courier New, monospace", size=18, color="Black"),
    yaxis_title="Frequency",
)
if not os.path.exists("images"):
    os.mkdir("images")
fig.write_image(os.path.join("images", "gpt_dist.png"))

# Figure 2: GPT vs Engel (2011)
# Enter data from Engel (2011)
engel_data_fig2 = {
    "All Studies": {
        "Allocation": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "Fraction": [
            0.3611,
            0.0914,
            0.0881,
            0.0891,
            0.0723,
            0.1674,
            0.0389,
            0.0197,
            0.0107,
            0.007,
            0.0544,
        ],
    },
    "One Shot Studies": {
        "Allocation": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "Fraction": [
            0.3133,
            0.0794,
            0.0934,
            0.0922,
            0.078,
            0.2127,
            0.0304,
            0.0142,
            0.0092,
            0.0064,
            0.0702,
        ],
    },
}
# Figure 2: GPT vs Engel et al. (2011)
df_engel = pd.DataFrame(engel_data_fig2["One Shot Studies"])
engel_mean = df_engel["Allocation"].dot(df_engel["Fraction"])
fig = px.bar(df_engel, x="Allocation", y="Fraction")
fig.update_layout(
    yaxis_range=[0, 0.4],
    xaxis=dict(tickmode="linear", tick0=0, dtick=0.1),
    font=dict(family="Courier New, monospace", size=18, color="Black"),
)
fig.add_vline(
    x=df.Allocation.mean() / 10, line_width=3, line_dash="dash", line_color="black"
)
fig.add_vline(x=engel_mean, line_width=3, line_dash="dash", line_color="red")
annotation1 = {
    "xref": "paper",
    "yref": "paper",
    "x": 0.325,
    "y": 0.8,
    "text": "Metastudy Mean",
    "showarrow": False,
    "font": {"size": 12, "color": "black"},
}
annotation2 = {
    "xref": "paper",
    "yref": "paper",
    "x": 0.53,
    "y": 0.90,
    "text": "GPT Mean",
    "showarrow": False,
    "font": {"size": 12, "color": "black"},
}
fig.update_layout({"annotations": [annotation1, annotation2]})
if not os.path.exists("images"):
    os.mkdir("images")
fig.write_image(os.path.join("images", "gpt_vs_engel.png"))


# Figure 3: plot GPT vs FHSS
# Read in data from Forsythe et al. (1994)
df_fhss = pd.read_stata(os.path.join("..", "data", "FHSS.DTA"))
df_fhss_dictator_without_pay = df_fhss[(df_fhss["pay"] == 0) & (df_fhss["exp"] == 1)]
df_fhss_dictator_with_pay = df_fhss[(df_fhss["pay"] == 1) & (df_fhss["exp"] == 1)]
df_fhss_collapsed = (
    df_fhss_dictator_without_pay.groupby("offer")["offer"]
    .count()
    .reset_index(name="Count")
)
df_fhss_collapsed["Count"] = (
    df_fhss_collapsed["Count"] / df_fhss_collapsed["Count"].sum()
)
df_fhss_collapsed.rename(
    columns={"offer": "Allocation", "Count": "Fraction"}, inplace=True
)
df_frac = df.groupby(["Allocation"])["Allocation"].count().reset_index(name="Count")
df_frac["GPT"] = df_frac["Count"] / df_frac["Count"].sum()
df_fhss_collapsed["Allocation"] *= 2  # scal to 10 as in our experiment
df_fhss_collapsed.rename(columns={"Fraction": "FHSS"}, inplace=True)
df_frac = df_frac.merge(df_fhss_collapsed, on="Allocation", how="outer")
df_frac.fillna(0, inplace=True)
# create plot of GPT results alongside Forsythe et al. (1994)
fig = px.bar(
    df_frac, x="Allocation", y=["GPT", "FHSS"], barmode="group", range_x=[0, 10]
)
fig.update_layout(
    xaxis=dict(tickmode="linear", tick0=0, dtick=1),
    font=dict(family="Courier New, monospace", size=18, color="Black"),
    yaxis_title="Frequency",
    legend_title="",
)
if not os.path.exists("images"):
    os.mkdir("images")
fig.write_image(os.path.join("images", "gpt_vs_fhss_nofair.png"))


# Table 1: GPT replicating Mengel (2018)
# Read in updated PD data
df = pd.read_csv(os.path.join("..", "data", "gpt_prisoner_results.csv"))
# keep just first 50 obs for each parameterization
# note that due to model interruptions, some parameterizations have more than 50 obs
df = df.groupby("model_parameterization").head(50).reset_index(drop=True)
# determine selected strategy
df["choice"] = df["model_answer"].str[-2]
df.loc[~df["choice"].isin(["A", "B"]), "choice"] = np.nan
phrases = [
    "choose option ",
    "choose choice ",
    "select choice ",
    " I might choose",
    "to choose ",
    "choosing option ",
    "I might choose choice ",
    "I would prefer option ",
    "select choice ",
    "select option",
    "the optimal strategy would be to choose ",
    "the optimal strategy is to choose ",
    "the dominant strategy would be to choose ",
]
for phrase in phrases:
    df["choice"] = np.where(
        df["choice"].isna,
        np.where(
            df["model_answer"].str.contains(phrase, case=False),
            df["model_answer"].str.split(phrase).str[-1],
            df["choice"],
        ),
        df["choice"],
    )
# remove spaces
df["choice"] = df["choice"].str.replace(" ", "")
# drop period at end
df.loc[:, "choice"] = df["choice"].str.replace(".", "", regex=False)
# keep just first character in string
df["choice"] = df["choice"].str[0]
# Set to nan if choice not clear
df.loc[
    df["model_answer"].str.contains("I can't make a choice", case=False), "choice"
] = np.nan
df.loc[~df["choice"].isin(["A", "B"]), "choice"] = np.nan
# Name strategies
df["Strategy"] = df["choice"].replace(
    {"A": "Cooperate", "B": "Defect", np.nan: "No Answer"}
)

# Frequency plot of strategies
fig = px.histogram(df, x="Strategy", histnorm="probability")
fig.update_layout(
    xaxis=dict(tickmode="linear", tick0=0, dtick=1),
    font=dict(family="Courier New, monospace", size=18, color="Black"),
    yaxis_title="Frequency",
)
if not os.path.exists("images"):
    os.mkdir("images")
fig.write_image(os.path.join("images", "gpt_pd_guo_responses.png"))

# drop non-answers
df = df[df["Strategy"] != "No Answer"]
print("Obs after dropping non-answers: ", df.shape[0])
# create indicators
df["Cooperate"] = np.where(df["Strategy"] == "Cooperate", 1, 0)
print("Overall cooperation rate = ", df["Cooperate"].mean())
# group by parameterization
df_studies = (
    df[["Cooperate", "model_parameterization"]]
    .groupby("model_parameterization")
    .mean()
    .reset_index()
)
# fill in values for a, b, c, d parameters of PD game
pd_parameterizations = {
    "a": [
        400,
        400,
        400,
        10,
        150,
        250,
        250,
        10,
        150,
        400,
        400,
        400,
        400,
        400,
        10,
        150,
        250,
        250,
        10,
        250,
        400,
        400,
    ],
    "b": [
        200,
        200,
        100,
        1,
        40,
        15,
        50,
        2,
        50,
        100,
        100,
        10,
        10,
        100,
        5,
        5,
        5,
        100,
        1,
        50,
        100,
        100,
    ],
    "c": [
        450,
        800,
        450,
        90,
        850,
        750,
        750,
        110,
        850,
        600,
        1200,
        450,
        800,
        450,
        90,
        850,
        750,
        750,
        110,
        750,
        600,
        1200,
    ],
    "d": [
        200,
        200,
        120,
        5,
        50,
        85,
        150,
        3,
        100,
        120,
        120,
        200,
        200,
        200,
        5,
        95,
        95,
        160,
        9,
        150,
        200,
        200,
    ],
}
df_studies["a"] = df_studies["model_parameterization"].apply(
    lambda x: pd_parameterizations["a"][x]
)
df_studies["b"] = df_studies["model_parameterization"].apply(
    lambda x: pd_parameterizations["b"][x]
)
df_studies["c"] = df_studies["model_parameterization"].apply(
    lambda x: pd_parameterizations["c"][x]
)
df_studies["d"] = df_studies["model_parameterization"].apply(
    lambda x: pd_parameterizations["d"][x]
)
# Compute measures of risk, temptation, and efficiency
df_studies["Risk"] = (df_studies["d"] - df_studies["b"]) / df_studies["d"]
df_studies["Temptation"] = (df_studies["c"] - df_studies["a"]) / df_studies["c"]
df_studies["Efficiency"] = (df_studies["a"] - df_studies["d"]) / df_studies["a"]


# Run regression for Table 1
exog_vars = ["Risk", "Temptation", "Efficiency"]
exog = sm.add_constant(df_studies[exog_vars])
model = sm.OLS(df_studies["Cooperate"], exog)
res = model.fit()

# Create Table 1
# create stars for table from GPT results
gpt_stars = {}
for i, v in enumerate(exog_vars + ["const"]):
    if res.pvalues[v] < 0.01:
        gpt_stars[v] = "***"
    elif res.pvalues[v] < 0.05:
        gpt_stars[v] = "**"
    elif res.pvalues[v] < 0.1:
        gpt_stars[v] = "*"
    else:
        gpt_stars[v] = ""
results_table = {
    "Variables": [
        "Risk",
        "",
        "Temptation",
        "",
        "Efficiency",
        "",
        "Constant",
        "",
        "Observations",
        "Sample",
        "R-squared",
    ],
    "Mengel (2018)": [
        "-0.269***",
        "(0.066)",
        "-0.055",
        "(0.096)",
        "0.308***",
        "(0.100)",
        "0.455***",
        "(0.098)",
        "45",
        "Lab/AMT",
        "0.484",
    ],
    "GPT-3.5": [
        str(round(res.params["Risk"], 3)) + gpt_stars["Risk"],
        "(" + str(round(res.bse["Risk"], 3)) + ")",
        str(round(res.params["Temptation"], 3)) + gpt_stars["Temptation"],
        "(" + str(round(res.bse["Temptation"], 3)) + ")",
        str(round(res.params["Efficiency"], 3)) + gpt_stars["Efficiency"],
        "(" + str(round(res.bse["Efficiency"], 3)) + ")",
        str(round(res.params["const"], 3)) + gpt_stars["const"],
        "(" + str(round(res.bse["const"], 3)) + ")",
        "22",
        "GPT",
        str(round(res.rsquared, 3)),
    ],
}

table_df = pd.DataFrame(results_table)
try:
    os.mkdir("tables")
except OSError:
    pass
table_df.to_latex(os.path.join("tables", "gpt_pd_reg_results.tex"), index=False)
