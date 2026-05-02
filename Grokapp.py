import streamlit as st
import pandas as pd
import sqlite3
from groq import Groq
import plotly.express as px
import re
import json

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide"
)

# ─── DB Connection ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_connection():
    return sqlite3.connect("data.db", check_same_thread=False)

conn = get_connection()

# ─── Groq Client ──────────────────────────────────────────────────────────────
@st.cache_resource
def get_groq_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

client = get_groq_client()

MODEL = "llama-3.3-70b-versatile"  # Free Groq model

# ─── Session State Init ────────────────────────────────────────────────────────
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "last_sql" not in st.session_state:
    st.session_state.last_sql = ""
if "table_info" not in st.session_state:
    st.session_state.table_info = []
if "dashboard_charts" not in st.session_state:
    st.session_state.dashboard_charts = []

# ─── Helper: Groq API Call ────────────────────────────────────────────────────
def groq_call(prompt: str, max_tokens: int = 500) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content

# ─── Helper: Clean SQL ─────────────────────────────────────────────────────────
def clean_sql(raw: str) -> str:
    raw = raw.replace("```sql", "").replace("```", "").strip()
    match = re.search(r"(SELECT\s.*?)(?:;|$)", raw, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else raw.strip()

# ─── Helper: Smart Chart ──────────────────────────────────────────────────────
def generate_smart_chart(result: pd.DataFrame, question: str):
    if result.empty or len(result.columns) < 2:
        return None, "Not enough columns to chart"

    chart_prompt = f"""
You are a data visualization expert.

User question: "{question}"

DataFrame columns: {list(result.columns)}
Column dtypes: {dict(result.dtypes.astype(str))}
Sample data:
{result.head(5).to_string()}

Choose the best chart type and columns. Return ONLY valid JSON, no explanation, no markdown:
{{
  "chart_type": "bar|line|pie|scatter|histogram|area",
  "x": "column_name",
  "y": "column_name_or_null",
  "color": "column_name_or_null",
  "title": "chart title",
  "reason": "one line why this chart type"
}}
"""
    try:
        raw = groq_call(chart_prompt, max_tokens=250)
        raw = raw.replace("```json", "").replace("```", "").strip()
        config = json.loads(raw)

        chart_map = {
            "bar":       px.bar,
            "line":      px.line,
            "area":      px.area,
            "scatter":   px.scatter,
            "histogram": px.histogram,
            "pie":       px.pie,
        }

        chart_type = config.get("chart_type", "bar")
        x_col      = config.get("x")
        y_col      = config.get("y")
        color_col  = config.get("color") if config.get("color") in result.columns else None
        title      = config.get("title", "Chart")
        reason     = config.get("reason", "")

        if x_col not in result.columns:
            x_col = result.columns[0]
        if y_col and y_col not in result.columns:
            numeric_cols = result.select_dtypes(include=["number"]).columns
            y_col = numeric_cols[0] if len(numeric_cols) > 0 else None

        fn = chart_map.get(chart_type, px.bar)

        if chart_type == "pie":
            fig = px.pie(result, names=x_col, values=y_col, title=title)
        elif chart_type == "histogram":
            fig = px.histogram(result, x=x_col, title=title, color=color_col)
        elif y_col:
            fig = fn(result, x=x_col, y=y_col, title=title, color=color_col)
        else:
            fig = px.bar(result, x=x_col, title=title)

        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(size=13)
        )
        return fig, reason

    except Exception as e:
        try:
            numeric_cols = result.select_dtypes(include=["number"]).columns
            if len(numeric_cols) >= 1:
                fig = px.bar(result, x=result.columns[0], y=numeric_cols[0],
                             title=f"{numeric_cols[0]} by {result.columns[0]}")
                return fig, "Fallback bar chart"
        except:
            pass
        return None, str(e)

# ─── Helper: Run SQL with auto-fix ─────────────────────────────────────────────
def run_sql(sql: str, tables_schema: str):
    try:
        result = pd.read_sql_query(sql, conn)
        return result, sql, None
    except Exception as e:
        st.warning("⚠️ Query failed. Auto-fixing...")
        fix_prompt = f"""
The following SQL query failed on SQLite:

{sql}

Error: {e}

Available tables:
{tables_schema}

Return ONLY the corrected SQL query. No explanation. No markdown.
"""
        try:
            fixed_sql = clean_sql(groq_call(fix_prompt, max_tokens=400))
            result = pd.read_sql_query(fixed_sql, conn)
            return result, fixed_sql, "fixed"
        except Exception as e2:
            return None, sql, str(e2)

# ─── Helper: Generate SQL with memory ─────────────────────────────────────────
def generate_sql(question: str, tables_schema: str) -> str:
    history_context = ""
    if st.session_state.conversation_history:
        last = st.session_state.conversation_history[-1]
        history_context = f"""
Previous question: "{last['question']}"
Previous SQL: {last['sql']}
"""

    prompt = f"""
You are a senior data analyst working with SQLite.

Available tables and columns:
{tables_schema}

{history_context}

If the new question is a follow-up (uses words like "now", "also", "instead", "filter that", "break down"),
modify the previous SQL accordingly.

New question: "{question}"

Return ONLY the SQL query. No explanation. No markdown. No backticks.
"""
    return clean_sql(groq_call(prompt, max_tokens=500))

# ─── Helper: Insights ──────────────────────────────────────────────────────────
def generate_insights(result: pd.DataFrame, question: str) -> str:
    prompt = f"""
You are a data analyst. The user asked: "{question}"

Here is the query result:
{result.head(10).to_string()}

Give 2-3 sharp, specific business insights based on the actual numbers.
Be concise. Use bullet points.
"""
    return groq_call(prompt, max_tokens=300)

# ─── Helper: SQL Explanation ───────────────────────────────────────────────────
def explain_sql(sql: str) -> str:
    return groq_call(
        f"Explain this SQL in simple plain English in 2-3 lines:\n\n{sql}",
        max_tokens=150
    )

# ══════════════════════════════════════════════════════════════════════════════
# UI STARTS HERE
# ══════════════════════════════════════════════════════════════════════════════

st.title("📊 AI Data Analyst")
st.caption("Powered by Groq (Free & Fast) — Upload CSVs → Ask questions → Get SQL, Charts & Insights")

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    st.success("🟢 Using Groq AI — Free & Fast")
    st.caption(f"Model: `{MODEL}`")

    mode = st.radio("Mode", ["💬 Single Query", "📊 Dashboard Builder"], index=0)

    st.divider()

    if st.session_state.conversation_history:
        st.subheader("🕓 Query History")
        for h in reversed(st.session_state.conversation_history[-5:]):
            st.caption(f"Q: {h['question'][:50]}...")

        if st.button("🗑️ Clear History"):
            st.session_state.conversation_history = []
            st.session_state.last_sql = ""
            st.rerun()

# ─── File Upload ───────────────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "Upload CSV files",
    type=["csv"],
    accept_multiple_files=True
)

if uploaded_files:
    table_info = []

    with st.expander("📁 Loaded Tables", expanded=True):
        for file in uploaded_files:
            try:
                df = pd.read_csv(file, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(file, encoding='latin1')
                    st.warning(f"{file.name}: Used latin1 encoding")
                except UnicodeDecodeError:
                    df = pd.read_csv(file, encoding='ISO-8859-1')
                    st.warning(f"{file.name}: Used ISO-8859-1 encoding")

            table_name = file.name.split(".")[0].replace(" ", "_").lower()
            df.to_sql(table_name, conn, if_exists="replace", index=False)

            cols = ", ".join(df.columns)
            table_info.append(f"{table_name} ({cols})")

            st.success(f"✅ {table_name} — {len(df)} rows, {len(df.columns)} columns")
            st.dataframe(df.head(3), use_container_width=True)

    st.session_state.table_info = table_info
    tables_schema = "\n".join(table_info)

    # ══════════════════════════════════════════════════════════════════════════
    # MODE 1: SINGLE QUERY
    # ══════════════════════════════════════════════════════════════════════════
    if mode == "💬 Single Query":

        st.subheader("💬 Ask a Question")

        if st.session_state.conversation_history:
            last_q = st.session_state.conversation_history[-1]['question']
            st.info(f"💡 Last question: *\"{last_q}\"* — You can ask a follow-up!")

        question = st.text_input(
            "Ask anything about your data:",
            placeholder="e.g. Show me monthly sales trend"
        )

        if question:
            with st.spinner("🤖 Groq is thinking..."):

                try:
                    sql_query = generate_sql(question, tables_schema)
                except Exception as e:
                    st.error(f"SQL generation failed: {e}")
                    st.stop()

                col1, col2 = st.columns([1, 1])

                with col1:
                    st.subheader("🔍 Generated SQL")
                    st.code(sql_query, language="sql")

                with col2:
                    st.subheader("📖 Explanation")
                    try:
                        explanation = explain_sql(sql_query)
                        st.write(explanation)
                    except:
                        st.warning("Could not generate explanation")

                result, final_sql, status = run_sql(sql_query, tables_schema)

                if status and status != "fixed":
                    st.error(f"Query failed: {status}")
                    st.stop()

                if status == "fixed":
                    st.success("✅ Query auto-fixed!")
                    st.code(final_sql, language="sql")

                if result is not None and not result.empty:

                    st.subheader("📋 Results")
                    st.dataframe(result, use_container_width=True)

                    st.subheader("📊 Smart Visualization")
                    with st.spinner("Choosing best chart..."):
                        fig, reason = generate_smart_chart(result, question)

                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption(f"💡 Chart chosen because: *{reason}*")
                    else:
                        st.warning(f"Could not generate chart: {reason}")

                    st.subheader("🧠 AI Insights")
                    with st.spinner("Generating insights..."):
                        try:
                            insights = generate_insights(result, question)
                            st.write(insights)
                        except:
                            st.warning("Could not generate insights")

                    st.session_state.conversation_history.append({
                        "question": question,
                        "sql": final_sql,
                        "result_summary": f"{len(result)} rows returned"
                    })
                    st.session_state.last_sql = final_sql

                elif result is not None and result.empty:
                    st.warning("Query returned no results. Try rephrasing your question.")

    # ══════════════════════════════════════════════════════════════════════════
    # MODE 2: DASHBOARD BUILDER
    # ══════════════════════════════════════════════════════════════════════════
    else:
        st.subheader("📊 Dashboard Builder")
        st.caption("Describe your dashboard and Groq will build multiple charts at once")

        dashboard_prompt = st.text_area(
            "Describe the dashboard you want:",
            placeholder="e.g. Show monthly sales trend, top 5 products by revenue, and sales by region",
            height=100
        )

        if st.button("🚀 Generate Dashboard", type="primary"):
            with st.spinner("🤖 Groq is planning your dashboard..."):

                split_prompt = f"""
You are a data analyst building a dashboard.

Available tables:
{tables_schema}

User wants this dashboard: "{dashboard_prompt}"

Break this into individual chart questions (max 4).
Return ONLY a valid JSON array, no markdown, no explanation:
[
  {{"question": "specific data question", "title": "Chart Title"}},
  {{"question": "specific data question", "title": "Chart Title"}}
]
"""
                try:
                    raw = groq_call(split_prompt, max_tokens=500)
                    raw = raw.replace("```json", "").replace("```", "").strip()
                    chart_questions = json.loads(raw)
                except Exception as e:
                    st.error(f"Could not plan dashboard: {e}")
                    st.stop()

            st.success(f"✅ Dashboard planned with {len(chart_questions)} charts!")

            st.session_state.dashboard_charts = []

            for i, cq in enumerate(chart_questions):
                q     = cq.get("question", "")
                title = cq.get("title", f"Chart {i+1}")

                with st.spinner(f"Building: {title}..."):
                    try:
                        sql    = generate_sql(q, tables_schema)
                        result, final_sql, status = run_sql(sql, tables_schema)

                        if result is not None and not result.empty:
                            fig, reason = generate_smart_chart(result, q)
                            if fig:
                                fig.update_layout(title=title)
                                st.session_state.dashboard_charts.append({
                                    "title":    title,
                                    "fig":      fig,
                                    "result":   result,
                                    "sql":      final_sql,
                                    "question": q
                                })
                    except Exception as e:
                        st.warning(f"Could not build '{title}': {e}")

        if st.session_state.dashboard_charts:
            st.divider()
            st.subheader("📊 Your Dashboard")

            charts = st.session_state.dashboard_charts

            for i in range(0, len(charts), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    if i + j < len(charts):
                        chart = charts[i + j]
                        with col:
                            st.plotly_chart(chart["fig"], use_container_width=True)
                            with st.expander("🔍 See SQL & Data"):
                                st.code(chart["sql"], language="sql")
                                st.dataframe(chart["result"], use_container_width=True)

            st.divider()
            st.subheader("🧠 Overall Dashboard Insights")
            with st.spinner("Generating overall insights..."):
                try:
                    all_data = "\n\n".join([
                        f"{c['title']}:\n{c['result'].head(5).to_string()}"
                        for c in charts
                    ])
                    overall_prompt = f"""
You are a senior data analyst.

Here is data from multiple charts in a dashboard:
{all_data}

Give 3-4 high-level business insights connecting patterns across these charts.
Be specific with numbers. Use bullet points.
"""
                    st.write(groq_call(overall_prompt, max_tokens=400))
                except:
                    st.warning("Could not generate overall insights")

else:
    st.info("👆 Upload one or more CSV files to get started")

    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 💬 Single Query Mode")
        st.caption("Ask questions in plain English. Get SQL, smart charts, and AI insights. Ask follow-up questions naturally.")
    with col2:
        st.markdown("### 📊 Dashboard Mode")
        st.caption("Describe a full dashboard in one sentence. Groq builds multiple charts automatically in a grid layout.")
    with col3:
        st.markdown("### ⚡ Powered by Groq")
        st.caption("Ultra-fast LLaMA 3.3 70B model. Completely free. Same features as the Claude version — no API costs.")