import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import itertools

def is_float_comma(s):
    try:
        float(s.replace(',', '.'))
        return True
    except ValueError:
        return False


def convert_single_digit(cell):
    """
    Convert single digits or float strings with comma to numeric type.
    To map the whole dataframe when it has been created
    """
    try:
        if isinstance(cell, str):
            cell_str = cell.strip()

            # Single digit integer
            if cell_str.isdigit() and len(cell_str) == 1:
                return int(cell_str)

            # Float with comma
            elif is_float_comma(cell_str):
                return float(cell_str.replace(',', '.'))

        return cell

    except (ValueError, TypeError):
        return cell


##############################
# Session state: groups (cached)
##############################

if "groups" not in st.session_state:
    st.session_state.groups = {}

st.title("SurWeys")
st.sidebar.header("Data & Grouping")

##############################
# Upload file
# conversion and mapping
##############################

file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
#skip_rows = st.sidebar.number_input("Skip first N lines", min_value=0, value=0, step=1)
col1, col2 = st.sidebar.columns(2)
with col1:
    skip_rows = st.number_input("Skip first N lines", min_value=0, value=0, step=1)
with col2:
    separator = st.text_input("Separator", value=';')


if file:
    try:
        df = pd.read_csv(file, skiprows=skip_rows, sep=separator)
        df = df.applymap(convert_single_digit) # Now all numbers are numbers
        st.sidebar.success(f"CSV loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    except Exception as e:
        st.sidebar.error(f"Error reading CSV: {e}")
        df = None
else:
    df = None


##########################################
# Group definition
# The user can select a filter for a group
# and the data to take from that filtering
# e.g: For all rows in which 'Gender'==1...
# ...take data from 'Height'
#########################################

if df is not None: # When data has been uploaded, show this

    # Filter column
    filter_col = st.sidebar.selectbox("Filter column", df.columns)

    # Operator
    operator = st.sidebar.selectbox("Operator", ["==", ">", "<", ">=", "<="])

    # Value
    if pd.api.types.is_numeric_dtype(df[filter_col]):
        value = st.sidebar.number_input("Value", value=0)
    else:
        value = st.sidebar.text_input("Value")

    # Column to extract data from
    data_col = st.sidebar.selectbox("Column to take data from", df.columns)


    # Group name
    group_name = st.sidebar.text_input( #automatically proposed name
        "Group name",
        f"{filter_col}{operator}{value}->{data_col}"
    )

    # Add group
    if st.sidebar.button("Add Group"):
        try:
            if operator == "==":
                group_df = df[df[filter_col] == value]
            elif operator == ">":
                group_df = df[df[filter_col] > value]
            elif operator == "<":
                group_df = df[df[filter_col] < value]
            elif operator == ">=":
                group_df = df[df[filter_col] >= value]
            elif operator == "<=":
                group_df = df[df[filter_col] <= value]

            if group_df.empty:
                st.sidebar.warning("This group is empty. No rows match the filter.")
            else:
                st.session_state.groups[group_name] = {
                    "data": group_df,
                    "plot_col": data_col
                }
                st.sidebar.success(
                    f"Group '{group_name}' added ({len(group_df)} rows)."
                )

        except Exception as e:
            st.sidebar.error(f"Error creating group: {e}")

    # List groups
    if st.session_state.groups:
        st.sidebar.subheader("CREATED GROUPS")
        for name, g in st.session_state.groups.items():
            st.sidebar.write(f"{name}: {len(g['data'])} rows")

    # Clear groups
    if st.sidebar.button("CLEAR GROUPS"):
        st.session_state.groups = {}
        st.sidebar.success("All groups cleared.")


    ######################
    # Plot settings
    ######################
    st.sidebar.subheader("Plot Settings")
    plot_type = st.sidebar.selectbox(
        "Plot type",
        ["Violin", "Box"]
    )

    palette = st.sidebar.selectbox(
        "Color palette",
        ["Set2", "Set1", "deep", "muted", "pastel", "dark", "colorblind"]
    )

##############################
# Custom plot labels
#############################

st.sidebar.subheader("Figure Labels")

custom_ylabel = st.sidebar.text_input(
    "Y-axis label",
    value="Values")

custom_title = st.sidebar.text_input(
    "Plot title",
    value="Distributions")

# Checkbox to show significance
show_significance = st.sidebar.checkbox(
    "Show significance (p-values)",
    value=True)


#########################
# PLOT SECTION
#########################
st.subheader("Fast exploration of numeric CSV datasets")
# dummy variable
plot_type="Violin"

if st.session_state.groups:# and len(st.session_state.groups) >= 2: Deprecated, we can also only plot 1

    long_df = pd.DataFrame(columns=["Value", "Group"])

    for name, group_info in st.session_state.groups.items():
        group_df = group_info["data"]
        plot_col = group_info["plot_col"]

        numeric_col = pd.to_numeric(group_df[plot_col], errors='coerce').dropna()

        if not numeric_col.empty:
            temp_df = pd.DataFrame({
                "Value": numeric_col,
                "Group": name # name given by the user
            })
            # Expand full dataset in each iteration
            long_df = pd.concat([long_df, temp_df], ignore_index=True)

    if not long_df.empty:

        fig, ax = plt.subplots()

        # -------- Plot selection --------
        if plot_type == "Violin":
            sns.violinplot(
                x="Group",
                y="Value",
                data=long_df,
                ax=ax,
                inner="quartile",
                palette=palette
            )
        else:
            sns.boxplot(
                x="Group",
                y="Value",
                data=long_df,
                ax=ax,
                palette=palette
            )

        ##############################
        # P-values and annotations
        ##############################

        groups = long_df["Group"].unique()
        group_data = {
                    g: long_df[long_df["Group"] == g]["Value"].astype(float).values
                    for g in groups}

        comparisons = list(itertools.combinations(groups, 2)) # Compare all against each other

        y_max = long_df["Value"].max()
        y_min = long_df["Value"].min()
        y_range = y_max - y_min if y_max > y_min else 1.0
        offset = y_range * 0.1 # Hardcoded this, should be fine
        current_height = y_max + offset
        if show_significance:
            for g1, g2 in comparisons:

                stat, p_value = stats.ttest_ind(
                    group_data[g1],
                    group_data[g2],
                    equal_var=False
                )

                x1 = list(groups).index(g1)
                x2 = list(groups).index(g2)

                ax.plot( # Plot the open bracket!
                    [x1, x1, x2, x2],
                    [current_height,
                     current_height + offset * 0.3,
                     current_height + offset * 0.3,
                     current_height],
                    lw=1,
                    c='grey' # always black
                )

                ax.text(
                    (x1 + x2) / 2,
                    current_height + offset * 0.3,
                    f"p = {p_value:.3e}",
                    ha="center",
                    va="bottom"
                )

                current_height += offset # each will be a little higher

        ax.set_ylabel(custom_ylabel)
        ax.set_title(custom_title)

        st.pyplot(fig)

    else:
        st.info("No numeric data to plot for the defined groups.")

elif st.session_state.groups:
    st.info("Add at one group to see a plot.")

else:
    st.info("Upload a CSV, define your interest groups and filter them with simple operations.  \nSurWeys automatically creates distribution plots for a fast data exploration.  \n* Now customizable with color palettes and integrated significante tests!!")
    st.info("Example:  \n* Filter column 'Age'<=35 → Extract 'Income'→ New group 'Junior Income'  \n* Filter column 'Age'>35 → Extract 'Income' → New group 'Senior Income' \n* A violin-plot distribution is automatically generated, that you can tweak later  \n* Define as many sub-groups as you want!")
