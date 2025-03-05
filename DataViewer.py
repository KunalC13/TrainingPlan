import streamlit as st
import pandas as pd

# Set the page layout to 'wide' so you have more horizontal space
st.set_page_config(layout="wide")

st.title("User Details Viewer")
st.write("Upload an Excel file and select a user to see their details at a glance.")

# 1. File uploader widget
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file:
    # 2. Read the Excel file into a DataFrame
    df = pd.read_excel(uploaded_file)

    # 3. Check if the "Name" column exists
    if "Name" in df.columns:
        # Create a dropdown of unique names
        unique_names = sorted(df["Name"].dropna().unique())
        selected_name = st.selectbox("Select a user:", unique_names)

        # Filter rows for the selected user
        user_details = df[df["Name"] == selected_name]

        if not user_details.empty:
            st.markdown(f"### Details for: **{selected_name}**")

            # ---------------------------
            # Option A: Vertical Layout
            # ---------------------------
            # This will show each column/value pair in a vertical list.
            st.write("**Vertical Layout:**")
            for idx, row in user_details.iterrows():
                st.write(f"**Record Index:** {idx}")
                # for col in user_details.columns:
                #     st.write(f"**{col}:** {row[col]}")
                # st.write("---")

            # ---------------------------
            # Option B: Transposed Table
            # ---------------------------
            # If you only want to see a single row, or if you don't mind
            # columns as rows, you can transpose the DataFrame.
            st.write("**Transposed Table Layout:**")
            # If multiple rows exist for the user, you'll see multiple columns
            # in the transposed table, one for each row.
            st.table(user_details.T)

        else:
            st.warning("No details found for the selected name.")
    else:
        st.error("The uploaded file does not contain a 'Name' column.")
else:
    st.info("Awaiting an Excel file upload.")
