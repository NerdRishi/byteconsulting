import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import datetime
from utils import get_metric_rules
import numpy as np
import base64
import json
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from typing import Union, Tuple  # Use older typing for Python < 3.10 compatibility
### NEW ###
import io
import os
from fpdf import FPDF

### END NEW ###


st.set_page_config(layout="centered", page_title="Analysis | Byte Consulting")
LOGO_PATH = "assets/logo.png"

# --- AUTHENTICATION SETUP ---
with open('./config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

name = st.session_state.get("name")
authentication_status = st.session_state.get("authentication_status")
username = st.session_state.get("username")


# --- CORRECTED: SMART CACHING & DATA LOADING FUNCTIONS ---
@st.cache_data(ttl=3600, show_spinner="Reading dataset from disk...")
def _load_parquet_from_disk(file_path_str: str, file_mod_time: float) -> pd.DataFrame:
    df = pd.read_parquet(file_path_str)
    df['Date'] = pd.to_datetime(df['Date'])
    return df


def load_active_data(username: str) -> Tuple[Union[pd.DataFrame, None], Union[str, None]]:
    data_source = st.session_state.get('data_source')
    if data_source == 'local':
        if 'local_df' in st.session_state:
            df = st.session_state.local_df.copy()
            df['Date'] = pd.to_datetime(df['Date'])
            return df, "Local File"
        return None, None
    elif data_source == 'saved':
        dataset_name = st.session_state.get('selected_dataset_name')
        if dataset_name:
            file_path = Path(f"data_cache/{username}/{dataset_name}.parquet")
            if file_path.exists():
                mod_time = file_path.stat().st_mtime
                df = _load_parquet_from_disk(str(file_path), mod_time)
                return df.copy(), dataset_name
            else:
                st.error(f"Error: The selected dataset '{dataset_name}' was not found.")
                return None, None
    return None, None


# --- Other helper functions ---
def load_json_file(file_path, default_data):
    if file_path.exists():
        with open(file_path, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return default_data
    return default_data


def save_json_file(data_list, file_path):
    with open(file_path, 'w') as f:
        json.dump(data_list, f, indent=4)


def format_value(value, metric_name, is_custom=False, display_as_percent=False):
    if pd.isna(value): return "N/A"
    if is_custom and display_as_percent: return f"{value :.2f}%"
    if is_custom:
        if isinstance(value, (int, np.integer)): return f"{value:,}"
        return f"{value:,.2f}"
    if '%' in metric_name and abs(value) <= 2: return f"{value :.2f}%"
    if '(Rs)' in metric_name or '₹' in metric_name or "Cost" in metric_name: return f"₹ {value:,.2f}"
    if isinstance(value, (int, np.integer)): return f"{value:,}"
    return f"{value:,.2f}"


def perform_calculation(num1, op, num2):
    if num1 is None or num2 is None or pd.isna(num1) or pd.isna(num2): return np.nan
    if op == '+': return num1 + num2
    if op == '-': return num1 - num2
    if op == '*': return num1 * num2
    if op == '/':
        if num2 == 0: return np.nan
        return num1 / num2
    return np.nan


@st.cache_data
def convert_df_to_csv(df):
    # This ensures the CSV export works well with the editor
    return df.to_csv(index=False).encode('utf-8')


### NEW: Function to convert DataFrame to a styled PDF ###

FONT_PATH = "assets/fonts/DejaVuSans.ttf"
@st.cache_data
def convert_df_to_pdf(df, brand_name, title):
    """
    Converts a DataFrame to a PDF file using fpdf2 and returns bytes.
    """

    class PDF(FPDF):
        def header(self):
            self.set_font("DejaVu", "B", 16)
            self.cell(0, 10, title, align='C', ln=1)
            self.set_font("DejaVu", "", 12)
            self.cell(0, 10, f"Brand: {brand_name}", align='C', ln=1)
            self.ln(4)

    pdf = PDF(orientation='L', unit='mm', format='A4')
    pdf.add_font('DejaVu', '', FONT_PATH, uni=True)
    pdf.add_font('DejaVu', 'B', FONT_PATH, uni=True)
    pdf.set_font("DejaVu", "", 10)
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Table settings
    col_widths = []
    usable_page_width = pdf.w - 2 * pdf.l_margin
    n_cols = len(df.columns)
    min_col_width = 30
    dynamic_width = max(min_col_width, usable_page_width / n_cols - 1)
    for col in df.columns:
        col_widths.append(dynamic_width)

    # Table header
    pdf.set_fill_color(242, 92, 54)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("DejaVu", "B", 10)
    for i, col in enumerate(df.columns):
        pdf.cell(col_widths[i], 8, str(col), border=1, align='C', fill=True)
    pdf.ln()
    pdf.set_font("DejaVu", "", 10)
    pdf.set_text_color(0, 0, 0)

    # Table rows
    fill = False
    for idx, row in df.iterrows():
        for i, value in enumerate(row):
            cell_text = str(value)
            pdf.cell(col_widths[i], 8, cell_text, border=1, align='C', fill=fill)
        pdf.ln()
        fill = not fill

    output = io.BytesIO()
    pdf.output(output)
    pdf_bytes = output.getvalue()
    output.close()
    return pdf_bytes

# --- MODIFIED: EPHEMERAL REPORT EDITOR FUNCTION ---
def render_editable_export_ui(summary_df: pd.DataFrame, key_prefix: str, filename: str, brand_name: str):
    import streamlit as st
    import pandas as pd

    st.markdown("---")
    show_editor_key = f"{key_prefix}_show_editor"
    dialog_df_key = f"{key_prefix}_dialog_df"

    _, col2, _ = st.columns([1, 2, 1])
    if col2.button("✏️ Edit & Export Report", key=f"{key_prefix}_edit_btn", use_container_width=True):
        st.session_state[show_editor_key] = True
        st.rerun()

    if st.session_state.get(show_editor_key, False):

        @st.dialog("Edit Report Before Exporting", width="large")
        def run_editor():
            st.info(
                "You can edit/delete rows and columns for this export. To hide a column in the export, deselect it below. The original data will not be changed.",
                icon="✍️"
            )

            # Always use summary_df as the base, never session_state for DataFrame
            edited_df = st.data_editor(
                summary_df.copy(),
                num_rows="dynamic",
                use_container_width=True,
                key=dialog_df_key
            )

            # COLUMN SELECTOR UI
            all_columns = list(edited_df.columns)
            selected_columns = st.multiselect(
                "Select columns to include in export (deselect to hide in export):",
                options=all_columns,
                default=all_columns,
                key=f"{key_prefix}_pdf_column_selector"
            )
            export_df = edited_df[selected_columns] if selected_columns else edited_df

            d1, d2, d3, d4 = st.columns([2, 2, 1, 3])

            d1.download_button(
                "📥 Download as CSV",
                convert_df_to_csv(export_df),
                filename,
                "text/csv",
                use_container_width=True,
                disabled=len(export_df.columns) == 0
            )

            pdf_bytes = convert_df_to_pdf(export_df, brand_name, "Performance Summary Report")
            d2.download_button(
                "📄 Download as PDF",
                pdf_bytes,
                filename.replace(".csv", ".pdf"),
                "application/pdf",
                use_container_width=True,
                type="primary",
                disabled=len(export_df.columns) == 0
            )

            if d4.button("Finish Editing", use_container_width=True):
                if show_editor_key in st.session_state:
                    del st.session_state[show_editor_key]
                st.rerun()
        run_editor()


# --- MAIN APP LOGIC (WRAPPED IN AUTHENTICATION CHECK) ---
if authentication_status:
    DATA_CACHE_DIR = Path(f"data_cache/{username}")
    CUSTOM_METRICS_FILE = DATA_CACHE_DIR / "custom_metrics.json"
    METRIC_PRESETS_FILE = DATA_CACHE_DIR / "metric_presets.json"

    if 'saved_custom_metrics' not in st.session_state:
        st.session_state.saved_custom_metrics = load_json_file(CUSTOM_METRICS_FILE, [])
    DEFAULT_PRESETS = {
        "Daily": ["Sales (Rs)", "Delivered orders", "Average order value (Rs)"],
        "Weekly": ["Sales (Rs)", "Delivered orders", "Bad orders", "Total complaints"],
        "Monthly": ["Sales (Rs)", "Impressions", "Menu opens", "Cart to orders (%)"]
    }
    if 'metric_presets' not in st.session_state:
        st.session_state.metric_presets = load_json_file(METRIC_PRESETS_FILE, DEFAULT_PRESETS)

    df_full, active_dataset_name = load_active_data(username)

    if df_full is None or df_full.empty:
        st.warning("⚠️ No dataset is currently selected or loaded.")
        st.info("Please go to the Data Management Hub to select a saved dataset or browse for a local file.")
        st.page_link("home.py", label="Go to Data Management Hub", icon="🏠")
        st.stop()

    min_date = df_full['Date'].dt.date.min()
    max_date = df_full['Date'].dt.date.max()
    metric_rules = get_metric_rules()
    all_available_metrics = sorted(df_full['Metric'].unique())

    with st.sidebar:
        st.write(f'Welcome *{name}*')
        authenticator.logout('Logout', 'sidebar', key='unique_logout_sidebar_dashboard')
        st.title("Navigation")
        st.page_link("home.py", label="Data Management Hub", icon="🏠")
        st.page_link("pages/Analysis_Dashboard.py", label="Analysis Dashboard", icon="📊")
        st.divider()

        if st.button("🔄 Refresh Data & Cache",
                     help="Clears the cache and reloads the latest version of the selected dataset from disk. Use this if you have just appended new data.",
                     use_container_width=True):
            st.cache_data.clear()
            st.toast("Data cache cleared. Reloading latest data...", icon="🔄")
            st.rerun()

        st.header("🔬 Analysis Controls")

        st.markdown("#### Filter by Hierarchy")
        selected_brand = st.selectbox("Select Brand", ["All Brands"] + sorted(df_full['Restaurant name'].unique()))
        with st.expander("Advanced Location & Outlet Filters", expanded=False):
            city_df = df_full[
                df_full['Restaurant name'] == selected_brand] if selected_brand != "All Brands" else df_full
            selected_city = st.selectbox("Select City", ["All Cities"] + sorted(city_df['City'].unique()))
            subzone_df = city_df[city_df['City'] == selected_city] if selected_city != "All Cities" else city_df
            selected_subzone = st.selectbox("Select Subzone", ["All Subzones"] + sorted(subzone_df['Subzone'].unique()))
            restaurant_df = subzone_df[
                subzone_df['Subzone'] == selected_subzone] if selected_subzone != "All Subzones" else subzone_df
            if restaurant_df is not None and not restaurant_df.empty:
                restaurant_df_display = restaurant_df.copy()
                restaurant_df_display['Res_Display'] = restaurant_df_display['Restaurant name'] + " (" + \
                                                       restaurant_df_display['Restaurant ID'].astype(str) + ")"
                selected_restaurants = st.multiselect("Select Individual Outlets",
                                                      sorted(restaurant_df_display['Res_Display'].unique()))
            else:
                selected_restaurants = []

        filtered_df = df_full.copy()
        if selected_brand != "All Brands": filtered_df = filtered_df[filtered_df['Restaurant name'] == selected_brand]
        if selected_city != "All Cities": filtered_df = filtered_df[filtered_df['City'] == selected_city]
        if selected_subzone != "All Subzones": filtered_df = filtered_df[filtered_df['Subzone'] == selected_subzone]
        if selected_restaurants:
            selected_ids = [int(res.split('(')[-1][:-1]) for res in selected_restaurants]
            filtered_df = filtered_df[filtered_df['Restaurant ID'].isin(selected_ids)]
        st.markdown("---")

        st.markdown("#### Analysis Options")


        def apply_preset():
            preset_name = st.session_state.metric_preset_selector
            if preset_name != "Custom":
                st.session_state.selected_metrics = [
                    m for m in st.session_state.metric_presets.get(preset_name, []) if m in all_available_metrics
                ]


        preset_selection = st.selectbox(
            "Metric Presets",
            ["Custom"] + list(st.session_state.metric_presets.keys()),
            key="metric_preset_selector",
            on_change=apply_preset
        )

        selected_metrics = st.multiselect(
            "Select Metrics to Analyze",
            all_available_metrics,
            key="selected_metrics"
        )

        with st.expander("⚙️ Manage Metric Presets"):
            preset_to_edit = st.selectbox("Choose a preset to modify", list(st.session_state.metric_presets.keys()))
            if preset_to_edit:
                new_metrics_for_preset = st.multiselect(
                    f"Update metrics for '{preset_to_edit}'",
                    all_available_metrics,
                    default=[m for m in st.session_state.metric_presets.get(preset_to_edit, []) if
                             m in all_available_metrics],
                    key=f"preset_edit_{preset_to_edit}"
                )
                if st.button(f"Save '{preset_to_edit}' Preset", use_container_width=True):
                    st.session_state.metric_presets[preset_to_edit] = new_metrics_for_preset
                    save_json_file(st.session_state.metric_presets, METRIC_PRESETS_FILE)
                    st.toast(f"Preset '{preset_to_edit}' updated!", icon="✅")
                    st.rerun()

        st.markdown("---")

        st.markdown("#### Date & Comparison")
        st.caption("Date ranges will automatically update after refreshing data.")
        is_monthly_view = st.checkbox("Show Monthly Summary", value=False)

        comparison_type = "Monthly"
        if not is_monthly_view:
            comparison_type = st.radio("Select Analysis Mode",
                                       ["Single Date", "Range", "Date vs Date", "Range vs Range"],
                                       index=1, horizontal=True, key="comparison_type_selector")

        p1_start, p1_end, p2_start, p2_end = None, None, None, None

        if comparison_type == "Single Date":
            p1_date = st.date_input("Select Date", max_date, min_value=min_date, max_value=max_date)
            p1_start, p1_end = p1_date, p1_date
        elif comparison_type == "Range":
            c1, c2 = st.columns(2)
            default_start = max_date - datetime.timedelta(days=6)
            p1_start = c1.date_input("Start Date", default_start if default_start >= min_date else min_date,
                                     min_value=min_date, max_value=max_date)
            p1_end = c2.date_input("End Date", max_date, min_value=p1_start, max_value=max_date)
        elif comparison_type == "Date vs Date":
            c1, c2 = st.columns(2)
            default_p1 = max_date - datetime.timedelta(days=7)
            p1_date = c1.date_input("Compare Date", default_p1 if default_p1 >= min_date else min_date,
                                    min_value=min_date, max_value=max_date)
            p2_date = c2.date_input("with Date", max_date, min_value=min_date, max_value=max_date)
            p1_start, p1_end, p2_start, p2_end = p1_date, p1_date, p2_date, p2_date
        elif comparison_type == "Range vs Range":
            st.markdown("**Period 1**")
            c1, c2 = st.columns(2)
            default_p1_end = max_date - datetime.timedelta(days=7)
            default_p1_start = default_p1_end - datetime.timedelta(days=6)
            p1_start = c1.date_input("Period 1 Start", default_p1_start if default_p1_start >= min_date else min_date,
                                     min_value=min_date, max_value=max_date)
            p1_end = c2.date_input("Period 1 End", default_p1_end if default_p1_end >= min_date else min_date,
                                   min_value=p1_start, max_value=max_date)
            st.markdown("**Period 2**")
            c3, c4 = st.columns(2)
            default_p2_start = max_date - datetime.timedelta(days=6)
            p2_start = c3.date_input("Period 2 Start", default_p2_start if default_p2_start >= min_date else min_date,
                                     min_value=min_date, max_value=max_date)
            p2_end = c4.date_input("Period 2 End", max_date, min_value=p2_start, max_value=max_date)

        if st.button("📊 Generate Analysis", type="primary", use_container_width=True):
            error = False
            if comparison_type != "Monthly":
                if p1_start and p1_end and p1_start > p1_end:
                    st.sidebar.error("Error: Start date cannot be after end date for Period 1.");
                    error = True
                elif p2_start and p2_end and p2_start > p2_end:
                    st.sidebar.error("Error: Start date cannot be after end date for Period 2.");
                    error = True

            if not error:
                st.session_state.analysis_generated = True
                st.session_state.comparison_type = comparison_type
                st.session_state.p1_start, st.session_state.p1_end = p1_start, p1_end
                st.session_state.p2_start, st.session_state.p2_end = p2_start, p2_end
                st.session_state.final_selected_metrics = selected_metrics
                st.session_state.filtered_df = filtered_df
                st.session_state.filter_selections = {
                    'brand': selected_brand, 'city': selected_city,
                    'subzone': selected_subzone, 'restaurants': selected_restaurants
                }
                for metric in all_available_metrics:
                    default_agg = metric_rules.get(metric, {'agg': 'sum'})['agg'].title()
                    if default_agg == "Mean": default_agg = "Average"
                    st.session_state[f"agg_{metric}"] = default_agg
                st.rerun()

        if st.session_state.get("analysis_generated", False):
            if st.button("🔄 Reset Analysis", use_container_width=True):
                keys_to_delete = [k for k in st.session_state.keys() if
                                  k.startswith('agg_') or k in ['analysis_generated', 'comparison_type',
                                                                'filter_selections', 'filtered_df',
                                                                'final_selected_metrics', 'selected_custom_metrics']]
                for key in keys_to_delete:
                    del st.session_state[key]
                st.rerun()

    # --- UI RENDERING (after sidebar) ---
    if Path(LOGO_PATH).exists():
        st.markdown(f"""
        <style>
            .logo-container {{ display: flex; justify-content: center; margin-bottom: 20px; }}
            .logo-img {{ transition: transform 0.4s ease; border-radius: 10px; }}
        </style>
        <div class="logo-container"><img src="data:image/png;base64,{base64.b64encode(open(LOGO_PATH, "rb").read()).decode()}" class="logo-img" width="350"></div>""",
                    unsafe_allow_html=True)

    st.markdown("""<style> h1 {{ color: #3A3A3A; text-align: center; }} h2, h3 {{ color: #F25C36; }} </style>""",
                unsafe_allow_html=True)

    st.title("Analysis Dashboard")
    st.markdown(
        f"<h5 style='text-align: center;'>Analyzing dataset: <strong><code>{active_dataset_name}</code></strong></h5>",
        unsafe_allow_html=True)

    if not st.session_state.get("analysis_generated", False):
        st.info("Adjust the controls in the sidebar and click 'Generate Analysis' to begin.")
        st.stop()

    # --- RETRIEVE SESSION STATE DATA ---
    df = st.session_state.filtered_df
    selections = st.session_state.filter_selections
    comparison_type = st.session_state.comparison_type

    title_parts = []
    if selections['restaurants']:
        title_parts.append(f"{len(selections['restaurants'])} selected outlets")
    elif selections['subzone'] != "All Subzones":
        title_parts.append(f"all outlets in {selections['subzone']}")
    elif selections['city'] != "All Cities":
        title_parts.append(f"all outlets in {selections['city']}")
    elif selections['brand'] != "All Brands":
        title_parts.append(f"all '{selections['brand']}' outlets")
    else:
        title_parts.append("All Restaurants")
    st.header(f"Analysis for: {', '.join(title_parts)}")

    # --- MONTHLY ANALYSIS LOGIC ---
    if comparison_type == "Monthly":
        summary_tab, trend_tab = st.tabs(["📊 Performance Summary", "📈 Trend & Contribution"])
        with summary_tab:
            st.subheader("Metric Aggregation Controls")
            num_metrics = len(st.session_state.final_selected_metrics)
            num_cols = min(num_metrics, 4)
            cols = st.columns(num_cols) if num_cols > 0 else []
            for i, metric in enumerate(st.session_state.final_selected_metrics):
                if cols:
                    with cols[i % num_cols]: st.radio(f"**{metric}**", options=['Sum', 'Average'], key=f"agg_{metric}",
                                                      horizontal=True)

            st.markdown("---")
            st.subheader("Select Custom Metrics to Display")
            selected_custom_metrics = st.multiselect("Choose from your saved custom metrics",
                                                     options=[m['name'] for m in st.session_state.saved_custom_metrics],
                                                     key="selected_custom_metrics")

            metrics_to_display = st.session_state.final_selected_metrics
            custom_name_to_def = {m['name']: m for m in st.session_state.saved_custom_metrics}
            custom_metrics_to_display = [custom_name_to_def[n] for n in selected_custom_metrics if
                                         n in custom_name_to_def]
            base_metrics_needed = set(metrics_to_display)
            for cm in custom_metrics_to_display:
                if cm['op1_type'] == 'metric': base_metrics_needed.add(cm['op1_val'])
                if cm['op2_type'] == 'metric': base_metrics_needed.add(cm['op2_val'])

            monthly_df = df[df['Metric'].isin(base_metrics_needed)].copy()
            monthly_df['Month'] = monthly_df['Date'].dt.to_period('M').astype(str)
            day_counts = monthly_df.groupby('Month')['Date'].nunique()

            monthly_agg = monthly_df.groupby(['Month', 'Metric'])['Value'].agg(['sum', 'mean']).reset_index()
            months = sorted(monthly_agg['Month'].unique())

            summary_data = []
            for metric in metrics_to_display:
                user_selection = st.session_state.get(f"agg_{metric}", "Sum")
                agg_choice = "mean" if user_selection == "Average" else "sum"
                metric_data = monthly_agg[monthly_agg['Metric'] == metric].set_index('Month')[agg_choice]
                metric_data.name = metric
                summary_data.append(metric_data)

            for cm in custom_metrics_to_display:
                if cm['op1_type'] == 'metric':
                    op1_data = monthly_agg[monthly_agg['Metric'] == cm['op1_val']].set_index('Month')['sum']
                elif cm['op1_type'] == 'number of days':
                    op1_data = pd.Series([day_counts.get(month, np.nan) for month in months], index=months)
                else:
                    op1_data = pd.Series(cm['op1_val'], index=months)
                if cm['op2_type'] == 'metric':
                    op2_data = monthly_agg[monthly_agg['Metric'] == cm['op2_val']].set_index('Month')['sum']
                elif cm['op2_type'] == 'number of days':
                    op2_data = pd.Series([day_counts.get(month, np.nan) for month in months], index=months)
                else:
                    op2_data = pd.Series(cm['op2_val'], index=months)
                aligned_op1, aligned_op2 = op1_data.align(op2_data, fill_value=np.nan)
                result_series = perform_calculation(aligned_op1, cm['op'], aligned_op2)
                if cm.get('display_as_percent', False): result_series *= 100
                result_series.name = f"{cm['name']}*"
                summary_data.append(result_series)

            st.markdown("---")
            st.subheader("Monthly Performance Table")
            if summary_data:
                summary_df = pd.concat(summary_data, axis=1).T

                if not summary_df.empty:
                    summary_df = summary_df.reindex(columns=months, fill_value=0)
                summary_df.columns = [f"{col} ({day_counts.get(col, 0)} days)" for col in summary_df.columns]

                def format_value_monthly(value, metric_name_with_star):
                    if pd.isna(value): return "N/A"
                    is_custom = "*" in str(metric_name_with_star)
                    metric_name = str(metric_name_with_star).replace("*", "")
                    if is_custom:
                        custom_def = custom_name_to_def.get(metric_name)
                        if custom_def and custom_def.get("display_as_percent", False): return f"{value:.2f}%"
                    if '%' in metric_name and abs(value) <= 2: return f"{value * 100:.2f}%"
                    if '(Rs)' in metric_name or '₹' in metric_name or "Cost" in metric_name: return f"₹ {value:,.2f}"
                    if float(value).is_integer(): return f"{int(value):,}"
                    return f"{value:,.2f}"

                formatted_df = summary_df.apply(lambda row: [format_value_monthly(val, row.name) for val in row], axis=1, result_type='expand')
                formatted_df.columns = summary_df.columns
                st.dataframe(formatted_df, use_container_width=True)

                ### MODIFIED: Pass the selected brand name to the export UI function ###
                render_editable_export_ui(
                    summary_df=formatted_df.reset_index().rename(columns={'index': 'Metric'}),
                    key_prefix='monthly',
                    filename='monthly_analysis_edited.csv',
                    brand_name=selections['brand']
                )
            else:
                st.warning("No metrics selected for analysis.")

        with trend_tab:
            st.subheader("Monthly Trend Analysis")
            trend_metric_options = st.session_state.final_selected_metrics + [f"{cm['name']}*" for cm in custom_metrics_to_display]
            trend_metric = st.selectbox("Select a metric to analyze its trend", trend_metric_options, key="monthly_trend_selector")
            if trend_metric and 'summary_df' in locals() and not summary_df.empty:
                if trend_metric in summary_df.index:
                    trend_df_data = summary_df.loc[trend_metric].T.reset_index()
                    trend_df_data.columns = ['Month', 'Value']
                    trend_df_data['Month'] = trend_df_data['Month'].apply(lambda x: x.split(' ')[0])
                    trend_df_data['Value'] = pd.to_numeric(trend_df_data['Value'], errors='coerce')
                    fig = px.line(trend_df_data, x='Month', y='Value', title=f"<b>Monthly Trend: {trend_metric.replace('*','')}</b>", markers=True)
                    fig.update_layout(hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)
                else: st.warning(f"Could not find data for '{trend_metric}'.")
            else: st.warning("Could not generate trend data. Please generate the summary table first.")

    # --- REGULAR ANALYSIS (NON-MONTHLY) ---
    else:
        p1_start, p1_end, p2_start, p2_end = st.session_state.p1_start, st.session_state.p1_end, st.session_state.p2_start, st.session_state.p2_end
        is_comparison = comparison_type in ["Date vs Date", "Range vs Range"]
        p1_df = df[(df['Date'].dt.date >= p1_start) & (df['Date'].dt.date <= p1_end)]
        p2_df = pd.DataFrame()
        if is_comparison: p2_df = df[(df['Date'].dt.date >= p2_start) & (df['Date'].dt.date <= p2_end)]

        summary_tab, trend_tab = st.tabs(["📊 Performance Summary", "📈 Trend & Contribution"])
        with summary_tab:
            st.subheader("Metric Aggregation Controls")
            num_metrics = len(st.session_state.final_selected_metrics)
            num_cols = min(num_metrics, 4)
            cols = st.columns(num_cols) if num_cols > 0 else []
            for i, metric in enumerate(st.session_state.final_selected_metrics):
                if cols:
                    with cols[i % num_cols]: st.radio(f"**{metric}**", options=['Sum', 'Average'], key=f"agg_{metric}", horizontal=True)

            with st.expander("🔧 Manage Custom Metrics"):
                with st.form("custom_metric_form", clear_on_submit=True):
                    new_metric_name = st.text_input("New Metric Name", placeholder="e.g., Ad Cost %")
                    operator = st.selectbox("Operator", ['/', '*', '+', '-'], index=0)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Operand A (Numerator)**")
                        op1_type = st.radio("Type for A", ["Metric", "Number", "Number of Days"], key="op1_type", horizontal=True)
                        op1_val = None
                        if op1_type == "Metric": op1_val = st.selectbox("Select Metric A", all_available_metrics, index=None, key="op1_metric_select")
                        elif op1_type == "Number": op1_val = st.number_input("Enter Number A", value=1.0, format="%.2f", key="op1_num_input")
                        else: op1_val = "num_days"
                    with c2:
                        st.markdown("**Operand B (Denominator)**")
                        op2_type = st.radio("Type for B", ["Metric", "Number", "Number of Days"], key="op2_type", horizontal=True)
                        op2_val = None
                        if op2_type == "Metric": op2_val = st.selectbox("Select Metric B", all_available_metrics, index=None, key="op2_metric_select")
                        elif op2_type == "Number": op2_val = st.number_input("Enter Number B", value=1.0, format="%.2f", key="op2_num_input")
                        else: op2_val = "num_days"
                    display_as_percent = st.checkbox("Display as % (multiply by 100)", key="display_as_percent_checkbox")
                    submitted = st.form_submit_button("💾 Save Custom Metric", use_container_width=True, type="primary")
                    if submitted:
                        if new_metric_name and op1_val is not None and op2_val is not None:
                            if new_metric_name in [m['name'] for m in st.session_state.saved_custom_metrics]:
                                st.error("A custom metric with this name already exists.")
                            else:
                                new_def = {'name': new_metric_name, 'op': operator, 'op1_type': op1_type.lower(), 'op1_val': op1_val, 'op2_type': op2_type.lower(), 'op2_val': op2_val, 'display_as_percent': display_as_percent}
                                st.session_state.saved_custom_metrics.append(new_def)
                                save_json_file(st.session_state.saved_custom_metrics, CUSTOM_METRICS_FILE)
                                st.toast(f"Saved custom metric '{new_metric_name}'!", icon="✅")
                                st.rerun()
                        else: st.warning("Please fill out all fields to save a custom metric.")
                st.markdown("---")
                st.write("**Your Saved Metrics:**")
                if not st.session_state.saved_custom_metrics: st.info("No custom metrics saved yet.")
                for i, metric_def in enumerate(st.session_state.saved_custom_metrics):
                    col1, col2 = st.columns([4, 1])
                    formula_op1 = f"'{metric_def['op1_val']}'" if metric_def['op1_type'] == 'metric' else metric_def[
                        'op1_val']
                    formula_op2 = f"'{metric_def['op2_val']}'" if metric_def['op2_type'] == 'metric' else metric_def[
                        'op2_val']
                    formula = f"({formula_op1}) {metric_def['op']} ({formula_op2})"
                    percent_hint = " [%]" if metric_def.get("display_as_percent", False) else ""
                    col1.text(f"• {metric_def['name']}{percent_hint} = {formula}")
                    if col2.button("❌", key=f"remove_metric_{i}", help="Remove metric"):
                        st.session_state.saved_custom_metrics.pop(i)
                        save_json_file(st.session_state.saved_custom_metrics, CUSTOM_METRICS_FILE)
                        st.rerun()

            st.markdown("---")
            st.subheader("Select Custom Metrics to Display")
            selected_custom_metrics = st.multiselect("Choose from your saved custom metrics",
                                                     options=[m['name'] for m in st.session_state.saved_custom_metrics],
                                                     key="selected_custom_metrics_standard")
            st.markdown("---")
            st.subheader("Performance Summary Table")
            table_data, calculated_values = [], {'Period 1': {}, 'Period 2': {}}
            custom_metrics_lookup = {cm['name']: cm for cm in st.session_state.saved_custom_metrics}
            custom_metrics_to_display = [custom_metrics_lookup[n] for n in selected_custom_metrics if
                                         n in custom_metrics_lookup]
            metrics_to_calc = set(st.session_state.final_selected_metrics)
            for cm in custom_metrics_to_display:
                if cm['op1_type'] == 'metric': metrics_to_calc.add(cm['op1_val'])
                if cm['op2_type'] == 'metric': metrics_to_calc.add(cm['op2_val'])

            for metric in metrics_to_calc:
                agg_method = st.session_state.get(f"agg_{metric}", "Sum")
                if agg_method == 'Sum':
                    value1 = p1_df[p1_df['Metric'] == metric]['Value'].sum()
                    value2 = p2_df[p2_df['Metric'] == metric]['Value'].sum() if is_comparison else 0
                else:  # Average
                    value1 = p1_df[p1_df['Metric'] == metric]['Value'].mean()
                    value2 = p2_df[p2_df['Metric'] == metric]['Value'].mean() if is_comparison else 0
                calculated_values['Period 1'][metric] = 0 if pd.isna(value1) else value1
                if is_comparison: calculated_values['Period 2'][metric] = 0 if pd.isna(value2) else value2

            for metric in st.session_state.final_selected_metrics:
                value1 = calculated_values['Period 1'].get(metric, 0)
                row_data = {"Metric": f"{metric} ({st.session_state.get(f'agg_{metric}', 'Sum')})"}
                if is_comparison:
                    value2 = calculated_values['Period 2'].get(metric, 0)
                    rule = metric_rules.get(metric, {'is_good_when_low': False})
                    delta = value2 - value1
                    p_change = (delta / abs(value1)) * 100 if value1 != 0 else float('inf') if delta > 0 else 0
                    status = "⚪️"
                    if delta != 0: status = "🟢" if (rule['is_good_when_low'] and delta < 0) or (
                                not rule['is_good_when_low'] and delta > 0) else "🔴"
                    row_data.update({"Status": status, "Period 1": format_value(value1, metric),
                                     "Period 2": format_value(value2, metric), "Change": format_value(delta, metric),
                                     "% Change": f"{p_change:+.1f}%" if value1 != 0 else "N/A"})
                else:
                    row_data["Value"] = format_value(value1, metric)
                table_data.append(row_data)

            period1_days = (p1_end - p1_start).days + 1 if p1_start and p1_end else 1
            period2_days = (p2_end - p2_start).days + 1 if is_comparison and p2_start and p2_end else 1
            for cm in custom_metrics_to_display:
                op1_val1 = calculated_values['Period 1'].get(cm['op1_val']) if cm[
                                                                                   'op1_type'] == 'metric' else period1_days if \
                cm['op1_type'] == 'number of days' else cm['op1_val']
                op2_val1 = calculated_values['Period 1'].get(cm['op2_val']) if cm[
                                                                                   'op2_type'] == 'metric' else period1_days if \
                cm['op2_type'] == 'number of days' else cm['op2_val']
                value1 = perform_calculation(op1_val1, cm['op'], op2_val1)
                if cm.get("display_as_percent", False) and value1 is not None and not pd.isna(value1): value1 *= 100
                row_data = {"Metric": f"{cm['name']}*"}
                if is_comparison:
                    op1_val2 = calculated_values['Period 2'].get(cm['op1_val']) if cm[
                                                                                       'op1_type'] == 'metric' else period2_days if \
                    cm['op1_type'] == 'number of days' else cm['op1_val']
                    op2_val2 = calculated_values['Period 2'].get(cm['op2_val']) if cm[
                                                                                       'op2_type'] == 'metric' else period2_days if \
                    cm['op2_type'] == 'number of days' else cm['op2_val']
                    value2 = perform_calculation(op1_val2, cm['op'], op2_val2)
                    if cm.get("display_as_percent", False) and value2 is not None and not pd.isna(value2): value2 *= 100
                    delta = (value2 - value1) if value1 is not None and value2 is not None and not (
                                pd.isna(value1) or pd.isna(value2)) else np.nan
                    p_change = (delta / abs(value1)) * 100 if value1 not in [None, 0] and not pd.isna(
                        value1) and not pd.isna(delta) else np.nan
                    row_data.update({"Status": "⚪️", "Period 1": format_value(value1, cm['name'], is_custom=True,
                                                                              display_as_percent=cm.get(
                                                                                  "display_as_percent")),
                                     "Period 2": format_value(value2, cm['name'], is_custom=True,
                                                              display_as_percent=cm.get("display_as_percent")),
                                     "Change": format_value(delta, cm['name'], is_custom=True,
                                                            display_as_percent=cm.get("display_as_percent")),
                                     "% Change": f"{p_change:+.1f}%" if not pd.isna(p_change) else "N/A"})
                else:
                    row_data["Value"] = format_value(value1, cm['name'], is_custom=True,
                                                     display_as_percent=cm.get("display_as_percent"))
                table_data.append(row_data)

            if table_data:
                summary_df = pd.DataFrame(table_data).rename(
                    columns={'Period 1': f"P1 ({p1_start} to {p1_end})" if p1_start != p1_end else f"P1 ({p1_start})",
                             'Period 2': f"P2 ({p2_start} to {p2_end})" if p2_start != p2_end else f"P2 ({p2_start})"})
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

                ### MODIFIED: Pass the selected brand name to the export UI function ###
                render_editable_export_ui(
                    summary_df=summary_df,
                    key_prefix='standard',
                    filename='analysis_summary_edited.csv',
                    brand_name=selections['brand']
                )

        with trend_tab:
            st.subheader("Metric Trend Analysis")
            st.info("Custom metrics are not available for trend analysis as they are aggregate calculations.")
            trend_metric = st.selectbox("Select a metric to analyze its trend", st.session_state.final_selected_metrics)
            if trend_metric:
                metric_p1_trend = p1_df[p1_df['Metric'] == trend_metric].groupby(p1_df['Date'].dt.date)[
                    'Value'].sum().reset_index()
                fig = go.Figure()
                p1_name = f'Period 1 ({p1_start} to {p1_end})' if p1_start != p1_end else f'Date: {p1_start}'
                fig.add_trace(go.Scatter(x=metric_p1_trend['Date'], y=metric_p1_trend['Value'], mode='lines+markers',
                                         name=p1_name, line=dict(color='royalblue')))
                if is_comparison:
                    metric_p2_trend = p2_df[p2_df['Metric'] == trend_metric].groupby(p2_df['Date'].dt.date)[
                        'Value'].sum().reset_index()
                    p2_name = f'Period 2 ({p2_start} to {p2_end})' if p2_start != p2_end else f'Date: {p2_start}'
                    fig.add_trace(
                        go.Scatter(x=metric_p2_trend['Date'], y=metric_p2_trend['Value'], mode='lines+markers',
                                   name=p2_name, line=dict(color='darkorange')))
                fig.update_layout(title_text=f"<b>Daily Trend: {trend_metric}</b>", hovermode="x unified",
                                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig, use_container_width=True)

                if is_comparison:
                    st.subheader(f"Contribution to Change in {trend_metric}")
                    breakdown_dim = None
                    if selections['brand'] == 'All Brands':
                        breakdown_dim = 'Restaurant name'
                    elif selections['city'] == 'All Cities':
                        breakdown_dim = 'City'
                    elif selections['subzone'] == 'All Subzones':
                        breakdown_dim = 'Subzone'
                    elif len(selections['restaurants']) > 1:
                        p1_df_copy = p1_df.copy();
                        p1_df_copy['Res_Display'] = p1_df_copy['Restaurant name'] + " (" + p1_df_copy[
                            'Restaurant ID'].astype(str) + ")"
                        p2_df_copy = p2_df.copy();
                        p2_df_copy['Res_Display'] = p2_df_copy['Restaurant name'] + " (" + p2_df_copy[
                            'Restaurant ID'].astype(str) + ")"
                        breakdown_dim = 'Res_Display'
                    if breakdown_dim:
                        df1_source = p1_df_copy if 'p1_df_copy' in locals() else p1_df
                        df2_source = p2_df_copy if 'p2_df_copy' in locals() else p2_df
                        p1_contrib = df1_source[df1_source['Metric'] == trend_metric].groupby(breakdown_dim)[
                            'Value'].sum()
                        p2_contrib = df2_source[df2_source['Metric'] == trend_metric].groupby(breakdown_dim)[
                            'Value'].sum()
                        contrib_df = pd.DataFrame({'Period 1': p1_contrib, 'Period 2': p2_contrib}).fillna(0)
                        contrib_df['Change'] = contrib_df['Period 2'] - contrib_df['Period 1']
                        contrib_df = contrib_df[contrib_df['Change'] != 0].sort_values('Change', ascending=False).head(
                            20)
                        if not contrib_df.empty:
                            contrib_fig = px.bar(contrib_df, x=contrib_df.index, y='Change', color='Change',
                                                 color_continuous_scale='RdYlGn',
                                                 labels={'x': breakdown_dim.replace('_', ' ').title(),
                                                         'Change': 'Contribution to Change'},
                                                 title=f"Top Contributors to Change in '{trend_metric}'")
                            contrib_fig.update_layout(coloraxis_showscale=False)
                            st.plotly_chart(contrib_fig, use_container_width=True)
                        else:
                            st.info(
                                f"No change detected in '{trend_metric}' at the {breakdown_dim.replace('_', ' ').title()} level.")
                    else:
                        st.info(
                            "Contribution analysis is available when aggregating multiple items (e.g., All Brands, All Cities, or multiple outlets).")
                else:
                    st.info(
                        "Contribution analysis is not applicable for a single period. Please select a comparison mode.")

# This part runs if the user is not logged in
elif authentication_status is False:
    st.error('Username/password is incorrect. Please return to the Home page to log in.')
    st.page_link("home.py", label="🏠 Go to Home Page")
elif authentication_status is None:
    st.warning('You must be logged in to access this page. Please go to the Home page to log in.')
    st.page_link("home.py", label="🏠 Go to Home Page")