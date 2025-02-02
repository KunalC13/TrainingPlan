import streamlit as st
import pandas as pd

# Set dark theme at the top
st.set_page_config(page_title="Training Plan Viewer", layout="wide", page_icon="🏋️")

# Function to load and clean CSV
@st.cache_data
def load_training_data(file_path):
    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip().str.replace("\n", " ").str.lower()
    return data

# Streamlit App
def main():
    st.title("🏋️ Training Plan Viewer")
    st.markdown("### 📄 Select a user to view their personalized training plan")

    # File Upload
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file:
        data = load_training_data(uploaded_file)

        # Select user
        user_names = data["name"].unique()
        selected_user = st.selectbox("🔍 Choose a User", user_names)

        if selected_user:
            user_data = data[data["name"] == selected_user]

            with st.expander(f"📖 **Training Plan for {selected_user}**", expanded=True):
                training_plan = user_data["training_plan"].values[0]
                # Color-enhanced Markdown formatting
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
