import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta



df_rl = pd.read_csv("./maskppo_24j.csv")


def to_datetime(initial_date: int, minutes: int) -> datetime:
    '''Converts minutes to datetime

    Args:
        initial_date (int): Initial date
        minutes (int): Minutes

    Returns:
        datetime: Datetime'''
    return initial_date.replace(hour=0, minute=0, second=0) + timedelta(minutes=minutes)


# Get today date at 00:00
initial_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)


# Rename personal column value
df_rl["machine"] = df_rl["MACHINE"].apply(lambda x: rf"m{x}")
df_rl["job"] = df_rl["ID"].apply(lambda x: rf"j{x}")

# Apply function to personal column 

# string to list
df_rl["personal"] = df_rl["OPERATORS"].apply(lambda x: len(x[1:-1].split(",")))

# apply to_datetime function to start_time and end_time
df_rl["start_time"] = df_rl["START TIME"].apply(
    lambda x: to_datetime(initial_date, x))
df_rl["end_time"] = df_rl["END TIME"].apply(lambda x: to_datetime(initial_date, x))

max_time = df_rl["end_time"].max()
start_time = df_rl["start_time"].min()
total_time = (max_time - start_time).total_seconds() / 60

operators = df_rl["personal"].sum()
total_jobs = df_rl["job"].nunique()

fig = px.timeline(df_rl,
                  x_start="start_time",
                  x_end="end_time",
                  title=f"SARL - Jobs: {total_jobs} - Timespan: {total_time} minutes - MOD: {operators}",
                  y="machine",
                  color="job",
                  hover_data=["job"],
                  template="seaborn",
                  color_discrete_sequence=px.colors.qualitative.Pastel,

                  labels={"machine": "Machine", "personal":"MOD", "job": "Job", "start_time": "Start time", "end_time": "End time"})

# Save figure as png
fig.write_image("./rl.png")
fig.show()