import streamlit as st
import pandas as pd


st.set_page_config(page_title="Training Plan Viewer", layout="wide", page_icon="🏋️")

st.markdown("""
    <style>
        /* Custom Background */
        .stApp {
            background-color: #0E1117;
            color: #EAEAEA;
        }
        
        /* Sidebar Customization */
        .stSidebar {
            background-color: #161A23;
        }

        /* Styling Download Button */
        div.stDownloadButton > button {
            background-color: #FFAA33 !important;
            color: black !important;
            border-radius: 10px;
            border: none;
            font-size: 16px;
        }
        
        div.stDownloadButton > button:hover {
            background-color: #FF8800 !important;
        }

        /* Styling Selectbox */
        .stSelectbox div[data-baseweb="select"] {
            background-color: #161A23 !important;
            color: #EAEAEA !important;
        }

        /* Expander Styling */
        .stExpander {
            background-color: #1E1E1E !important;
            border-radius: 10px !important;
        }

        /* Scrollbar Customization */
        ::-webkit-scrollbar {
            width: 8px;
        }

        ::-webkit-scrollbar-thumb {
            background: #FFAA33;
            border-radius: 10px;
        }

        ::-webkit-scrollbar-track {
            background: #1E1E1E;
        }

        /* Markdown Styling */
        pre {
            background-color: #161A23 !important;
            border-radius: 10px !important;
            padding: 15px;
        }
    </style>
""", unsafe_allow_html=True)


# Set Streamlit Page Config

# Function to load and clean CSV
@st.cache_data
def load_training_data(file_path):
    data = pd.read_csv(file_path)
    # Strip spaces, replace line breaks
    data.columns = data.columns.str.strip().str.replace("\n", " ", regex=True)
    return data

# Streamlit App
def main():
    st.title("🏋️ Training Plan Viewer")
    st.markdown("### 📄 Select a user to view their details and training plan")

    # File Upload
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file:
        data = load_training_data(uploaded_file)

        # Get exact column names from CSV
        column_names = list(data.columns)

        # Select user
        user_names = data[column_names[1]].unique()  # 'Name ' (with space)
        selected_user = st.selectbox("🔍 Choose a User", user_names)

        if selected_user:
            user_data = data[data[column_names[1]] == selected_user].iloc[0]  # Extract row as Series

            # Display User Details (All except 'Timestamp', 'Name', 'Training_Plan')
            st.subheader(f"👤 User Details: {selected_user}")
            excluded_columns = ["Timestamp", column_names[1], "Training_Plan"]
            user_details = user_data.drop(labels=excluded_columns)

            # Convert NaN values to 'Not Provided'
            user_details = user_details.fillna("Not Provided")

            # Define Styling for Better Readability
            detail_card_style = """
                <div style="
                    background-color: #1E1E1E;
                    padding: 15px;
                    margin-bottom: 10px;
                    border-radius: 10px;
                    border-left: 5px solid #FFAA33;">
                <p style="color: #FFAA33; font-size: 16px; margin: 0px;"><b>{question}</b></p>
                <p style="color: #EAEAEA; font-size: 16px; margin: 0px;">{answer}</p>
                </div>
            """

            # Display Details in Styled Cards
            user_details_html = ""
            for key, value in user_details.items():
                user_details_html += detail_card_style.format(question=key, answer=value)

            st.markdown(user_details_html, unsafe_allow_html=True)

            st.divider()  # Adds a horizontal divider line

            # Display Training Plan
            with st.expander(f"📖 **Training Plan for {selected_user}**", expanded=True):
                training_plan = user_data["Training_Plan"]
                formatted_plan = f"""
                <div style="background-color:#161A23;padding:15px;border-radius:10px;">
                <pre style="color:#EAEAEA;font-size:16px;">{training_plan}</pre>
                </div>
                """
                st.markdown(formatted_plan, unsafe_allow_html=True)

            # Download button
            st.download_button(
                label="📥 Download Training Plan",
                data=training_plan,
                file_name=f"{selected_user}_training_plan.txt",
                mime="text/plain",
                key=f"download_{selected_user}"
            )

if __name__ == "__main__":
    main()
