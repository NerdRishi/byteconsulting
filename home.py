import streamlit as st
import pandas as pd
from pathlib import Path
import datetime
import base64
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

st.set_page_config(layout="centered", page_title="Data Management | Byte Consulting")
LOGO_PATH = "assets/logo.png"

# --- NEW: Constant for the browse option in the dropdown ---
BROWSE_OPTION = "⬇️ Browse your own..."

# --- AUTHENTICATION SETUP ---
with open('./config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# --- LOGIN WIDGET ---
authenticator.login()

name = st.session_state.get("name")
authentication_status = st.session_state.get("authentication_status")
username = st.session_state.get("username")

if authentication_status:
    DATA_CACHE_DIR = Path(f"data_cache/{username}")
    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def get_saved_datasets():
        """Finds all saved .parquet files in the user's directory."""
        return sorted([f.stem for f in DATA_CACHE_DIR.glob("*.parquet")], reverse=True)

    # --- NEW: Callback function to manage state when the dataset selector changes ---
    def handle_dataset_selection():
        """
        Manages session state based on the main dataset selection.
        This determines if the app should use a 'saved' or 'local' data source.
        """
        selection = st.session_state.dataset_selector
        if selection == BROWSE_OPTION:
            st.session_state.data_source = 'local'
            # Clear other state variables to avoid conflicts
            if 'selected_dataset_name' in st.session_state:
                del st.session_state['selected_dataset_name']
        else:
            st.session_state.data_source = 'saved'
            st.session_state.selected_dataset_name = selection
            # Clear local dataframe if user switches back to a saved one
            if 'local_df' in st.session_state:
                del st.session_state['local_df']

    # This function remains unchanged, but is included for completeness
    def process_and_save_csv(uploaded_file, dataset_name, action='overwrite', append_to=None):
        save_path = DATA_CACHE_DIR / f"{dataset_name}.parquet"
        try:
            new_df = pd.read_csv(uploaded_file)
            expected_cols = ['Restaurant ID', 'Restaurant name', 'Subzone', 'City', 'Overview', 'Metric', 'Attribute', 'Value']
            if not all(col in new_df.columns for col in expected_cols):
                return False, f"CSV format error! Missing one or more required columns: {expected_cols}"
            new_df['Date'] = pd.to_datetime(new_df['Attribute'], errors='coerce')
            if new_df['Date'].isnull().any():
                return False, "Date format error! Could not parse some dates. Please use YYYY-MM-DD, DD-MM-YYYY, or DD-Mon-YYYY."
            new_df['Value'] = pd.to_numeric(new_df['Value'], errors='coerce')
            new_df.dropna(subset=['Value', 'Date'], inplace=True)

            if action == 'append_and_create_new' and append_to:
                existing_path = DATA_CACHE_DIR / f"{append_to}.parquet"
                if not existing_path.exists():
                    return False, f"Selected dataset {append_to} does not exist."
                existing_df = pd.read_parquet(existing_path)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df.drop_duplicates(subset=['Restaurant ID', 'Metric', 'Date'], keep='last', inplace=True)
                final_df = combined_df
                success_message = f"✅ Successfully created new dataset: **`{dataset_name}.parquet`** by appending to `{append_to}.parquet`"
            else:
                final_df = new_df
                success_message = f"✅ Successfully created new dataset: **`{dataset_name}.parquet`**"
            final_df.to_parquet(save_path)
            return True, success_message
        except Exception as e:
            return False, f"An unexpected error occurred: {e}"


    # --- SIDEBAR ---
    with st.sidebar:
        st.write(f'Welcome *{name}*')
        authenticator.logout('Logout', 'sidebar')
        st.title("Navigation")
        st.page_link("home.py", label="Data Management Hub", icon="🏠")
        st.page_link("pages/Analysis_Dashboard.py", label="Analysis Dashboard", icon="📊")

    # --- MAIN PAGE UI ---
    if Path(LOGO_PATH).exists():
        logo_html = f"""
        <style>
            .logo-container {{
                display: flex;
                justify-content: center;
                margin-bottom: 5px;
                margin-top: -60px;
            }}
            .logo-img {{
                transition: transform 0.4s ease, box-shadow 0.4s ease;
                border-radius: 10px;
                cursor: pointer;
                box-shadow: 0px 8px 15px rgba(0, 0, 0, 0.1);
            }}
            .logo-img:hover {{
                transform: scale(1.03) perspective(1000px) rotateY(-5deg);
                box-shadow: 0px 12px 25px rgba(242, 92, 54, 0.2);
            }}
            section.main > div:first-child {{ padding-top: 0rem !important; }}
        </style>
        <div class="logo-container">
            <img src="data:image/png;base64,{base64.b64encode(open(LOGO_PATH, "rb").read()).decode()}" class="logo-img" width="350">
        </div>
        """
        st.markdown(logo_html, unsafe_allow_html=True)

    st.markdown("""<style>h1 {{ color: #3A3A3A; text-align: center; }} h2, h3 {{ color: #F25C36; }} </style>""",
                unsafe_allow_html=True)

    st.title("Aggregator Growth Analytics Hub")
    st.header("1. Data Management")
    st.markdown("Select a saved dataset, browse a local file, or upload a new CSV file below.")

    st.subheader("Select Dataset for Analysis")
    saved_datasets = get_saved_datasets()

    # --- MODIFIED: Add the "Browse" option to the list of choices ---
    display_options = [BROWSE_OPTION] + saved_datasets

    # --- MODIFIED: More robust logic to determine the default selection ---
    # Set default data source if it doesn't exist in the session
    if 'data_source' not in st.session_state:
        st.session_state.data_source = 'saved' if saved_datasets else 'local'
        if st.session_state.data_source == 'saved':
            st.session_state.selected_dataset_name = saved_datasets[0]

    # Determine the current selection to set the index of the selectbox
    current_selection = BROWSE_OPTION if st.session_state.data_source == 'local' else st.session_state.get('selected_dataset_name')
    try:
        current_index = display_options.index(current_selection)
    except (ValueError, TypeError):
        current_index = 0 # Default to the first option if something is wrong

    st.selectbox(
        "Choose one of your saved datasets, or browse for a local file:",
        display_options,
        index=current_index,
        key="dataset_selector",
        on_change=handle_dataset_selection, # Use the new callback function
        help="This selection will be used across all analysis pages. Local files are not saved."
    )

    # --- NEW: Conditional UI for browsing a local file ---
    if st.session_state.get('data_source') == 'local':
        st.info("You've chosen to use a local file. This file will not be saved to your account.", icon="ℹ️")
        local_parquet_file = st.file_uploader(
            "Upload a .parquet file from your computer",
            type=['parquet'],
            key='local_file_uploader'
        )
        if local_parquet_file:
            try:
                # Read the file into a DataFrame and store it in session_state
                df = pd.read_parquet(local_parquet_file)
                # Simple validation of the parquet file's columns
                expected_cols = ['Restaurant ID', 'Metric', 'Date', 'Value']
                if all(col in df.columns for col in expected_cols):
                    st.session_state.local_df = df
                    st.success(f"✅ Loaded local file **`{local_parquet_file.name}`**. Go to the **Analysis Dashboard** to explore.")
                else:
                    st.error(f"The selected Parquet file is missing required columns. Please ensure it contains at least: {expected_cols}")
                    if 'local_df' in st.session_state:
                        del st.session_state['local_df']
            except Exception as e:
                st.error(f"Could not read or validate the Parquet file. Error: {e}")
                if 'local_df' in st.session_state:
                    del st.session_state['local_df']

    # --- MODIFIED: Show status message only when a saved dataset is active ---
    elif st.session_state.get('data_source') == 'saved' and st.session_state.get("selected_dataset_name"):
        st.markdown(
            f"Selected saved dataset: **`{st.session_state.selected_dataset_name}.parquet`**. Go to the **Analysis Dashboard** to explore.")
    elif not saved_datasets and st.session_state.data_source != 'local':
        st.warning("No datasets found for your account. Please upload a new CSV file below to begin.")


    # --- The "Upload New CSV" section is functionally unchanged, but with corrected state logic ---
    st.markdown("---")
    st.subheader("Upload New CSV File")

    with st.expander("Click here to create a new dataset from a CSV file", expanded=False):
        # Your notification state management is fine
        if "notification_visible" not in st.session_state: st.session_state.notification_visible = False
        if "notification_message" not in st.session_state: st.session_state.notification_message = ""
        if "notification_type" not in st.session_state: st.session_state.notification_type = "success"

        def dismiss_notification(): st.session_state.notification_visible = False

        if not st.session_state.notification_visible:
            with st.form(key="csv_upload_form", clear_on_submit=True):
                uploaded_file = st.file_uploader("Upload your CSV data file", type=['csv'])
                action_choice = st.radio(
                    "What would you like to do?",
                    ("Create a new dataset", "Append to an existing dataset"),
                    horizontal=True, key="action_choice", disabled=(len(saved_datasets) == 0)
                )
                action = 'overwrite' if action_choice == "Create a new dataset" else 'append_and_create_new'

                if action == 'overwrite':
                    dataset_name_input = st.text_input("Enter a name for the new dataset", f"dataset_{datetime.date.today().strftime('%Y_%m_%d')}")
                else: # append
                    append_to_dataset_name = st.selectbox("Choose a dataset to append to:", saved_datasets)
                    st.info("A new dataset will be created by appending the CSV to the selected dataset, preserving both the original and the new one.")
                    dataset_name_input = st.text_input("Enter a unique name for the new combined dataset", f"{append_to_dataset_name}_appended_{datetime.date.today().strftime('%Y_%m_%d')}")

                submitted = st.form_submit_button("Process and Save File", type="primary", use_container_width=True)
                if submitted:
                    dataset_name = dataset_name_input.strip()
                    append_target = append_to_dataset_name if action == 'append_and_create_new' else None

                    if not uploaded_file: st.warning("Please provide a file to upload.")
                    elif not dataset_name: st.warning("Please provide a dataset name.")
                    elif dataset_name in saved_datasets: st.warning(f"A dataset called '{dataset_name}' already exists. Please use a new name.")
                    else:
                        with st.spinner("Processing your file..."):
                            success, msg = process_and_save_csv(uploaded_file, dataset_name, action=action, append_to=append_target)

                        st.session_state.notification_message = msg
                        st.session_state.notification_type = "success" if success else "error"
                        st.session_state.notification_visible = True
                        if success:
                            # CRITICAL: On success, make the new dataset the active one
                            st.session_state.data_source = 'saved'
                            st.session_state.selected_dataset_name = dataset_name
                            if 'local_df' in st.session_state: del st.session_state['local_df']
                        st.rerun()
        else: # Display notification
            # This part of your code for displaying notifications is fine and remains unchanged
            notif_type = st.session_state.notification_type
            msg = st.session_state.notification_message
            if st.button("Dismiss Notification", key="dismiss_notif_btn"):
                dismiss_notification()
                st.rerun()
            if notif_type == 'success': st.success(msg)
            else: st.error(msg)


elif authentication_status is False:
    st.error('Username/password is incorrect')
elif authentication_status is None:
    st.warning('Please enter your username and password to access the app.')