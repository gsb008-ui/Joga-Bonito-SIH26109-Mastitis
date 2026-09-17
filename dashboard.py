import streamlit as st
import sqlite3
import requests
import time
import random
import pandas as pd
from datetime import date, timedelta

from database import create_database


# ============================================================
# DATABASE
# ============================================================

create_database()

DATABASE_NAME = "mastitis.db"


def get_connection():
    return sqlite3.connect(DATABASE_NAME)


def ensure_demo_cow():
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT OR IGNORE INTO cows
        (cow_tag, breed, age, lactation_num)
        VALUES (?, ?, ?, ?)
        """,
        ("Demo Cow 123", "Jersey", 5, 3)
    )

    connection.commit()
    connection.close()


ensure_demo_cow()


def get_cows():
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT cow_tag, breed, age, lactation_num
        FROM cows
        ORDER BY cow_tag
        """
    )

    cows = cursor.fetchall()
    connection.close()

    return cows


def get_history(cow_tag, limit=7):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            record_date,
            days_since_calving,
            prior_mastitis,
            milk_yield,
            body_temp,
            milk_conductivity,
            activity_score
        FROM daily_records
        WHERE cow_tag = ?
        ORDER BY record_date DESC
        LIMIT ?
        """,
        (cow_tag, limit)
    )

    history = cursor.fetchall()
    connection.close()

    return history


def save_daily_record(
    cow_tag,
    record_date,
    days_since_calving,
    prior_mastitis,
    milk_yield,
    body_temp,
    milk_conductivity,
    activity_score
):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT OR REPLACE INTO daily_records (
            cow_tag,
            record_date,
            days_since_calving,
            prior_mastitis,
            milk_yield,
            body_temp,
            milk_conductivity,
            activity_score
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            cow_tag,
            record_date,
            days_since_calving,
            prior_mastitis,
            milk_yield,
            body_temp,
            milk_conductivity,
            activity_score
        )
    )

    connection.commit()
    connection.close()


def delete_daily_record(cow_tag, record_date):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        DELETE FROM daily_records
        WHERE cow_tag = ? AND record_date = ?
        """,
        (cow_tag, record_date)
    )

    connection.commit()
    connection.close()


def delete_all_records(cow_tag):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        DELETE FROM daily_records
        WHERE cow_tag = ?
        """,
        (cow_tag,)
    )

    connection.commit()
    connection.close()


def delete_cow(cow_tag):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        "DELETE FROM daily_records WHERE cow_tag = ?",
        (cow_tag,)
    )

    cursor.execute(
        "DELETE FROM cows WHERE cow_tag = ?",
        (cow_tag,)
    )

    connection.commit()
    connection.close()


# ============================================================
# AI HELPERS
# ============================================================

def make_model_input(cow_info, history):
    """
    Convert the latest health history into the exact
    20-feature structure expected by the trained model.
    """

    chronological = list(reversed(history))
    latest = chronological[-1]

    milk_values = [row[3] for row in chronological]
    temp_values = [row[4] for row in chronological]
    conductivity_values = [row[5] for row in chronological]
    activity_values = [row[6] for row in chronological]

    milk_change = milk_values[-1] - milk_values[0]
    temp_change = temp_values[-1] - temp_values[0]
    conductivity_change = (
        conductivity_values[-1]
        - conductivity_values[0]
    )
    activity_change = (
        activity_values[-1]
        - activity_values[0]
    )

    count = max(len(chronological) - 1, 1)

    return {
        "age": cow_info[2],
        "lactation_num": cow_info[3],
        "days_since_calving": latest[1],
        "prior_mastitis": latest[2],

        "milk_yield": latest[3],
        "body_temp": latest[4],
        "milk_conductivity": latest[5],
        "activity_score": latest[6],

        "milk_yield_7d_mean":
            sum(milk_values) / len(milk_values),

        "milk_yield_change_7d":
            milk_change,

        "body_temp_7d_mean":
            sum(temp_values) / len(temp_values),

        "body_temp_change_7d":
            temp_change,

        "milk_conductivity_7d_mean":
            sum(conductivity_values)
            / len(conductivity_values),

        "milk_conductivity_change_7d":
            conductivity_change,

        "activity_score_7d_mean":
            sum(activity_values)
            / len(activity_values),

        "activity_score_change_7d":
            activity_change,

        "milk_yield_slope_7d":
            milk_change / count,

        "body_temp_slope_7d":
            temp_change / count,

        "milk_conductivity_slope_7d":
            conductivity_change / count,

        "activity_score_slope_7d":
            activity_change / count
    }


def get_ai_probability(cow_info, history):

    if len(history) < 2:
        return None

    try:

        payload = make_model_input(
            cow_info,
            history
        )

        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=payload,
            timeout=10
        )

        if response.status_code != 200:
            return None

        result = response.json()

        return float(
            result["risk_probability"]
        )

    except Exception:
        return None


def urgency_label(score):

    if score >= 75:
        return "HIGH"

    if score >= 50:
        return "ELEVATED"

    if score >= 25:
        return "WATCH"

    return "NORMAL"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Mastitis Early Warning",
    page_icon="🐄",
    layout="centered"
)


# ============================================================
# DESIGN
# ============================================================

st.markdown(
    """
    <style>

    html, body, [class*="css"] {
        font-family: "Times New Roman", Times, serif !important;
    }

    .stApp {
        background-color: #c7e3e9;
    }

    section[data-testid="stSidebar"] {
        background-color: #aed6df;
    }

    h1, h2, h3 {
        font-family: "Times New Roman", Times, serif !important;
        color: #083d48 !important;
    }

    p, label {
        color: #123f49 !important;
        font-size: 1.08rem !important;
    }

    [data-testid="stCaptionContainer"] {
        color: #164b56 !important;
        font-size: 1rem !important;
    }

    div[data-testid="stMetric"] {
        background-color: #e8f6f8;
        border-radius: 10px;
        padding: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🐄 Mastitis AI")

st.sidebar.caption(
    "Dairy health early-warning system"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Farm Dashboard",
        "🐄 Cow Health",
        "📅 Daily Health Check",
        "🎬 7-Day Demo",
        "➕ Register Cow",
        "⚙️ Manage Cows"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "SIH 2026 Prototype"
)


# ============================================================
# FARM DASHBOARD
# ============================================================

if page == "🏠 Farm Dashboard":

    st.title("🐄 Farm Dashboard")

    st.caption(
        "Monitor the herd and focus attention where it matters."
    )

    cows = get_cows()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Cows",
            len(cows)
        )

    with col2:
        st.metric(
            "Today",
            date.today().strftime("%d %b")
        )

    with col3:
        st.metric(
            "AI Status",
            "Ready"
        )

    st.divider()

    st.subheader("Herd Overview")

    table_rows = []

    for cow in cows:

        tag, breed, age, lactation = cow

        history = get_history(
            tag,
            7
        )

        if history:

            latest = history[0]

            probability = get_ai_probability(
                cow,
                history
            )

            if probability is None:

                inspection = "Awaiting data"

            else:

                inspection = urgency_label(
                    probability * 100
                )

            table_rows.append(
                {
                    "Cow": tag,
                    "Breed": breed,
                    "Age": age,
                    "Lactation": lactation,
                    "Milk (L)": round(
                        latest[3],
                        1
                    ),
                    "Temp °C": round(
                        latest[4],
                        2
                    ),
                    "Activity": round(
                        latest[6],
                        1
                    ),
                    "Inspection": inspection
                }
            )

        else:

            table_rows.append(
                {
                    "Cow": tag,
                    "Breed": breed,
                    "Age": age,
                    "Lactation": lactation,
                    "Milk (L)": "—",
                    "Temp °C": "—",
                    "Activity": "—",
                    "Inspection": "No data"
                }
            )

    if table_rows:

        dashboard_df = pd.DataFrame(
            table_rows
        )

        st.dataframe(
            dashboard_df,
            use_container_width=True,
            hide_index=True,
            height=min(
                420,
                90 + 42 * len(table_rows)
            )
        )

    st.caption(
        "Inspection status is a decision-support signal, "
        "not a veterinary diagnosis."
    )


# ============================================================
# COW HEALTH
# ============================================================

elif page == "🐄 Cow Health":

    st.title("🐄 Cow Health")

    cows = get_cows()

    if not cows:

        st.warning(
            "Register a cow first."
        )

        st.stop()

    selected = st.selectbox(
        "Select cow",
        [cow[0] for cow in cows]
    )

    cow_info = next(
        cow for cow in cows
        if cow[0] == selected
    )

    st.caption(
        f"{cow_info[1]} • "
        f"{cow_info[2]} years • "
        f"Lactation {cow_info[3]}"
    )

    history = get_history(
        selected,
        7
    )

    if not history:

        st.info(
            "No health records available yet."
        )

        st.stop()

    st.subheader(
        "7-Day Health History"
    )

    table_rows = []

    for record in reversed(history):

        table_rows.append(
            {
                "Date": record[0],
                "Milk (L)": round(
                    record[3],
                    1
                ),
                "Temp °C": round(
                    record[4],
                    2
                ),
                "Conductivity": round(
                    record[5],
                    2
                ),
                "Activity": round(
                    record[6],
                    1
                )
            }
        )

    st.dataframe(
        pd.DataFrame(table_rows),
        use_container_width=True,
        hide_index=True,
        height=300
    )

    st.divider()

    probability = get_ai_probability(
        cow_info,
        history
    )

    if probability is not None:

        score = probability * 100

        st.subheader(
            "Inspection Priority"
        )

        st.progress(
            min(score / 100, 1.0)
        )

        st.write(
            f"**Urgency to inspect: "
            f"{score:.0f}/100**"
        )

        st.caption(
            "This is a trend-based decision-support signal. "
            "It does not mean the cow has mastitis."
        )

        if score >= 75:

            st.error(
                "🔴 HIGH — inspect promptly"
            )

        elif score >= 50:

            st.warning(
                "🟠 ELEVATED — inspect and monitor closely"
            )

        elif score >= 25:

            st.info(
                "🟡 WATCH — continue monitoring"
            )

        else:

            st.success(
                "🟢 NORMAL — no immediate inspection priority"
            )


# ============================================================
# DAILY HEALTH CHECK
# ============================================================

elif page == "📅 Daily Health Check":

    st.title(
        "📅 Daily Health Check"
    )

    cows = get_cows()

    if not cows:

        st.warning(
            "Register a cow first."
        )

        st.stop()

    selected = st.selectbox(
        "Select cow",
        [cow[0] for cow in cows]
    )

    selected_date = st.date_input(
        "Health check date",
        value=date.today(),
        min_value=date.today(),
        max_value=date.today()
        + timedelta(days=365)
    )

    if selected_date > date.today():

        st.info(
            "Future date selected. "
            "Health information cannot be entered "
            "until that date."
        )

    else:

        milk = st.number_input(
            "Milk produced today (litres)",
            min_value=0.0,
            value=15.0
        )

        st.caption(
            "অসমীয়া: আজি গৰুৱে কিমান গাখীৰ দিছে।"
        )

        temperature = st.number_input(
            "Body temperature (°C)",
            min_value=35.0,
            max_value=43.0,
            value=38.5
        )

        st.caption(
            "অসমীয়া: গৰুৰ শৰীৰৰ উষ্ণতা।"
        )

        activity = st.number_input(
            "Activity level",
            min_value=0.0,
            value=70.0
        )

        st.caption(
            "অসমীয়া: গৰুটো কিমান সক্ৰিয় হৈ আছে।"
        )

        conductivity = st.number_input(
            "Milk conductivity",
            min_value=0.0,
            value=7.5
        )

        st.caption(
            "অসমীয়া: গাখীৰৰ গুণগত পৰিৱৰ্তনৰ এটা জোখ।"
        )

        if st.button(
            "💾 Save Health Check",
            type="primary"
        ):

            save_daily_record(
                selected,
                str(selected_date),
                100,
                0,
                milk,
                temperature,
                conductivity,
                activity
            )

            st.success(
                f"✅ Health check saved for {selected}."
            )


# ============================================================
# 7-DAY CURTAIN RAISER
# ============================================================

elif page == "🎬 7-Day Demo":

    st.title(
        "🎬 7-Day Curtain Raiser"
    )

    st.caption(
        "Watch how changing health signals influence "
        "inspection priority over time."
    )

    st.info(
        "🎥 DEMONSTRATION MODE — "
        "The seven days use randomized simulated data. "
        "The demonstration can produce healthy, moderate, "
        "or concerning trajectories."
    )

    st.subheader(
        "🐄 Meet Demo Cow 123"
    )

    st.write(
        "**Jersey • 5 years • Lactation 3**"
    )

    st.write(
        "We follow the cow for seven simulated days, "
        "observe the changing signals, and then ask the "
        "AI to interpret the overall pattern."
    )

    st.divider()

    if st.button(
        "▶️ Start Randomized 7-Day Demo",
        type="primary"
    ):

        # ----------------------------------------------------
        # RANDOM SCENARIO
        # ----------------------------------------------------

        scenario = random.choice(
            [
                "Healthy",
                "Moderate",
                "Concerning"
            ]
        )

        # Random baseline.
        base_milk = random.uniform(
            16.0,
            20.0
        )

        base_temp = random.uniform(
            38.1,
            38.5
        )

        base_conductivity = random.uniform(
            6.0,
            7.0
        )

        base_activity = random.uniform(
            78,
            95
        )

        # Different possible trajectories.
        if scenario == "Healthy":

            milk_drift = random.uniform(
                -0.10,
                0.12
            )

            temp_drift = random.uniform(
                -0.015,
                0.015
            )

            conductivity_drift = random.uniform(
                -0.03,
                0.04
            )

            activity_drift = random.uniform(
                -0.3,
                0.3
            )

        elif scenario == "Moderate":

            milk_drift = random.uniform(
                -0.35,
                -0.08
            )

            temp_drift = random.uniform(
                0.02,
                0.07
            )

            conductivity_drift = random.uniform(
                0.05,
                0.16
            )

            activity_drift = random.uniform(
                -1.4,
                -0.3
            )

        else:

            milk_drift = random.uniform(
                -0.70,
                -0.30
            )

            temp_drift = random.uniform(
                0.05,
                0.14
            )

            conductivity_drift = random.uniform(
                0.12,
                0.28
            )

            activity_drift = random.uniform(
                -3.5,
                -1.0
            )

        # Clear previous demo.
        delete_all_records(
            "Demo Cow 123"
        )

        start_date = (
            date.today()
            - timedelta(days=6)
        )

        progress = st.progress(0)

        status = st.empty()

        day_display = st.empty()

        metric_display = st.empty()

        daily_values = []

        # ----------------------------------------------------
        # SEVEN-DAY CURTAIN
        # ----------------------------------------------------

        for day_number in range(1, 8):

            milk = (
                base_milk
                + milk_drift
                * (day_number - 1)
                + random.uniform(
                    -0.25,
                    0.25
                )
            )

            temperature = (
                base_temp
                + temp_drift
                * (day_number - 1)
                + random.uniform(
                    -0.04,
                    0.04
                )
            )

            conductivity = (
                base_conductivity
                + conductivity_drift
                * (day_number - 1)
                + random.uniform(
                    -0.10,
                    0.10
                )
            )

            activity = (
                base_activity
                + activity_drift
                * (day_number - 1)
                + random.uniform(
                    -2.5,
                    2.5
                )
            )

            milk = round(
                max(milk, 5),
                2
            )

            temperature = round(
                temperature,
                2
            )

            conductivity = round(
                max(conductivity, 3),
                2
            )

            activity = round(
                max(activity, 10),
                1
            )

            simulation_date = (
                start_date
                + timedelta(
                    days=day_number - 1
                )
            )

            save_daily_record(
                "Demo Cow 123",
                str(simulation_date),
                100 + day_number,
                0,
                milk,
                temperature,
                conductivity,
                activity
            )

            daily_values.append(
                {
                    "Day": day_number,
                    "Date": str(simulation_date),
                    "Milk": milk,
                    "Temperature": temperature,
                    "Conductivity": conductivity,
                    "Activity": activity
                }
            )

            status.info(
                f"📅 Day {day_number} of 7 — "
                f"collecting health signals..."
            )

            day_display.subheader(
                f"Day {day_number}"
            )

            metric_display.write(
                f"🥛 Milk: **{milk} L**   |   "
                f"🌡️ Temperature: **{temperature}°C**   |   "
                f"🏃 Activity: **{activity}**   |   "
                f"🥛 Conductivity: **{conductivity}**"
            )

            progress.progress(
                day_number / 7
            )

            time.sleep(
                0.9
            )

        status.success(
            f"✅ Seven days complete — "
            f"randomized trajectory: {scenario}"
        )

        time.sleep(
            0.8
        )

        st.divider()

        # ----------------------------------------------------
        # CALCULATE DAILY INSPECTION URGENCY
        # ----------------------------------------------------

        st.subheader(
            "📈 Inspection Urgency Trend"
        )

        st.caption(
            "The graph shows how the model's signal changes "
            "as more daily information becomes available."
        )

        urgency_values = []

        demo_info = (
            "Demo Cow 123",
            "Jersey",
            5,
            3
        )

        for day_number in range(1, 8):

            records = get_history(
                "Demo Cow 123",
                limit=day_number
            )

            if len(records) >= 2:

                probability = get_ai_probability(
                    demo_info,
                    records
                )

                if probability is None:
                    score = 0
                else:
                    score = probability * 100

            else:

                score = 0

            score = max(
                0,
                min(
                    score,
                    100
                )
            )

            urgency_values.append(
                round(score, 1)
            )

        chart_df = pd.DataFrame(
            {
                "Day": range(1, 8),
                "Inspection Urgency": urgency_values
            }
        )

        st.line_chart(
            chart_df.set_index("Day"),
            y="Inspection Urgency",
            height=300
        )

        # ----------------------------------------------------
        # SCALE
        # ----------------------------------------------------

        st.write(
            "**Inspection priority scale**"
        )

        scale = pd.DataFrame(
            {
                "Range": [
                    "0–25",
                    "25–50",
                    "50–75",
                    "75–100"
                ],
                "Meaning": [
                    "Normal",
                    "Watch",
                    "Elevated",
                    "High priority"
                ]
            }
        )

        st.dataframe(
            scale,
            use_container_width=True,
            hide_index=True,
            height=190
        )

        # ----------------------------------------------------
        # FINAL AI REVEAL
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "🤖 AI Interpretation"
        )

        with st.spinner(
            "AI is interpreting the seven-day trend..."
        ):

            time.sleep(
                1.3
            )

        final_score = urgency_values[-1]

        st.metric(
            "Urgency to Inspect",
            f"{final_score:.0f}/100"
        )

        if final_score >= 75:

            st.error(
                "🔴 HIGH INSPECTION PRIORITY"
            )

            st.write(
                "The combined trend signals suggest that "
                "this cow deserves prompt inspection."
            )

        elif final_score >= 50:

            st.warning(
                "🟠 ELEVATED INSPECTION PRIORITY"
            )

            st.write(
                "The pattern is becoming more concerning. "
                "The cow should be monitored closely."
            )

        elif final_score >= 25:

            st.info(
                "🟡 WATCH"
            )

            st.write(
                "Some changes are visible. "
                "Continue monitoring the cow."
            )

        else:

            st.success(
                "🟢 NORMAL"
            )

            st.write(
                "The simulated health pattern remains "
                "relatively stable."
            )

        st.caption(
            "Important: this is a prototype decision-support "
            "signal based on simulated data. It does not diagnose "
            "mastitis and does not replace veterinary examination."
        )


# ============================================================
# REGISTER COW
# ============================================================

elif page == "➕ Register Cow":

    st.title(
        "➕ Register New Cow"
    )

    st.caption(
        "Register each cow once before recording daily health data."
    )

    with st.form(
        "register_cow_form",
        clear_on_submit=True
    ):

        cow_tag = st.text_input(
            "Cow Tag",
            placeholder="Example: A130"
        )

        breed = st.text_input(
            "Breed",
            placeholder="Example: Jersey"
        )

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=20,
            value=5
        )

        lactation = st.number_input(
            "Lactation Number",
            min_value=1,
            max_value=10,
            value=1
        )

        submitted = st.form_submit_button(
            "➕ Register Cow",
            type="primary"
        )

    if submitted:

        if not cow_tag.strip():

            st.warning(
                "Please enter a cow tag."
            )

        elif not breed.strip():

            st.warning(
                "Please enter the breed."
            )

        else:

            connection = get_connection()
            cursor = connection.cursor()

            try:

                cursor.execute(
                    """
                    INSERT INTO cows
                    (cow_tag, breed, age, lactation_num)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        cow_tag.strip(),
                        breed.strip(),
                        age,
                        lactation
                    )
                )

                connection.commit()
                connection.close()

                st.success(
                    f"✅ Cow {cow_tag} registered successfully!"
                )

            except sqlite3.IntegrityError:

                connection.close()

                st.error(
                    "⚠️ A cow with this tag already exists."
                )


# ============================================================
# MANAGE COWS
# ============================================================

elif page == "⚙️ Manage Cows":

    st.title(
        "⚙️ Manage Cows"
    )

    cows = get_cows()

    if not cows:

        st.info(
            "No cows registered."
        )

    else:

        management_rows = []

        for cow in cows:

            tag, breed, age, lactation = cow

            history = get_history(
                tag,
                30
            )

            management_rows.append(
                {
                    "Cow": tag,
                    "Breed": breed,
                    "Age": age,
                    "Lactation": lactation,
                    "Records": len(history),
                    "Type": (
                        "Demo"
                        if tag == "Demo Cow 123"
                        else "Farm Cow"
                    )
                }
            )

        st.subheader(
            "Registered Cows"
        )

        st.dataframe(
            pd.DataFrame(
                management_rows
            ),
            use_container_width=True,
            hide_index=True,
            height=min(
                400,
                90 + 42 * len(management_rows)
            )
        )

        st.divider()

        selected = st.selectbox(
            "Select cow to manage",
            [
                cow[0]
                for cow in cows
            ]
        )

        if selected == "Demo Cow 123":

            st.info(
                "Demo Cow 123 is protected so the presentation "
                "demo remains available."
            )

            if st.button(
                "🔄 Reset Demo Simulation"
            ):

                delete_all_records(
                    selected
                )

                st.success(
                    "✅ Demo simulation data reset."
                )

                st.rerun()

        else:

            history = get_history(
                selected,
                30
            )

            if history:

                st.subheader(
                    "Daily Records"
                )

                record_rows = []

                for record in history:

                    record_rows.append(
                        {
                            "Date": record[0],
                            "Milk (L)": round(
                                record[3],
                                1
                            ),
                            "Temp °C": round(
                                record[4],
                                2
                            ),
                            "Conductivity": round(
                                record[5],
                                2
                            ),
                            "Activity": round(
                                record[6],
                                1
                            )
                        }
                    )

                st.dataframe(
                    pd.DataFrame(
                        record_rows
                    ),
                    use_container_width=True,
                    hide_index=True,
                    height=min(
                        300,
                        90 + 42 * len(record_rows)
                    )
                )

                past_dates = [
                    row[0]
                    for row in history
                    if date.fromisoformat(
                        row[0]
                    ) <= date.today()
                ]

                if past_dates:

                    record_to_delete = st.selectbox(
                        "Select a past/present record to delete",
                        past_dates
                    )

                    if st.button(
                        "🗑️ Delete Selected Record"
                    ):

                        delete_daily_record(
                            selected,
                            record_to_delete
                        )

                        st.success(
                            f"✅ Record {record_to_delete} deleted."
                        )

                        st.rerun()

            else:

                st.info(
                    "This cow has no daily records."
                )

            st.divider()

            confirm = st.checkbox(
                f"I understand that deleting {selected} "
                "will remove the cow and all its records."
            )

            if confirm:

                if st.button(
                    "🗑️ Delete Cow Permanently"
                ):

                    delete_cow(
                        selected
                    )

                    st.success(
                        f"✅ {selected} deleted successfully."
                    )

                    st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "SIH 2026 Prototype • "
    "AI-Based Predictive Modelling for Early Forecasting "
    "of Bovine Mastitis"
)