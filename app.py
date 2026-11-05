from pathlib import Path
from datetime import date
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# OLIST INSIGHTS - Interactive E-Commerce Business Dashboard
# Place this file in the same folder as the Olist CSV files.
# Run: streamlit run app.py
# ============================================================

st.set_page_config(
    page_title="Olist Insights | Business Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR  # CSV files are expected beside app.py


# --------------------------- Styling ---------------------------

st.markdown("""
<style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e7eaf0;
        padding: 15px 16px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(20,35,60,.04);
    }
    [data-testid="stMetricLabel"] {font-size: .88rem;}
    [data-testid="stMetricValue"] {font-size: 1.65rem;}
    .section-note {color: #667085; font-size: .92rem;}
    div[data-testid="stSidebar"] {background: #f7f8fb;}
</style>
""", unsafe_allow_html=True)


# --------------------------- Data loading ---------------------------

FILES = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "products": "olist_products_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "translation": "product_category_name_translation.csv",
}


@st.cache_data(show_spinner="Loading Olist data...")
def load_csvs(folder_string):
    folder = Path(folder_string)
    result = {}
    missing = []
    for key, filename in FILES.items():
        path = folder / filename
        if path.exists():
            result[key] = pd.read_csv(path, low_memory=False)
        else:
            result[key] = pd.DataFrame()
            missing.append(filename)
    return result, missing


def parse_dates(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def safe_divide(a, b):
    return float(a / b) if b not in (0, None) and pd.notna(b) else 0.0


def money(value):
    return f"R$ {value:,.2f}"


def pct(value):
    return f"{value:.1f}%"


def empty_message(message):
    st.info(message)


def download_csv(df, filename, label="Download CSV"):
    if df is None or df.empty:
        st.caption("No rows available to export for the current selection.")
        return
    data = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(label, data, filename, "text/csv", key=f"download_{filename}")


def style_chart(fig, height=360):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=45, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", color="#344054"),
        legend_title_text="",
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#edf0f5", zeroline=False)
    return fig


@st.cache_data(show_spinner=False)
def prepare_data(folder_string):
    raw, missing = load_csvs(folder_string)

    orders = raw["orders"].copy()
    items = raw["items"].copy()
    customers = raw["customers"].copy()
    products = raw["products"].copy()
    payments = raw["payments"].copy()
    reviews = raw["reviews"].copy()
    sellers = raw["sellers"].copy()
    translation = raw["translation"].copy()

    # Parse date columns
    orders = parse_dates(orders, [
        "order_purchase_timestamp", "order_approved_at",
        "order_delivered_carrier_date", "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ])
    if not items.empty and "shipping_limit_date" in items:
        items = parse_dates(items, ["shipping_limit_date"])
    if not reviews.empty and "review_creation_date" in reviews:
        reviews = parse_dates(reviews, ["review_creation_date", "review_answer_timestamp"])

    # Basic type cleanup
    if not items.empty:
        for col in ["price", "freight_value", "order_item_id"]:
            if col in items:
                items[col] = pd.to_numeric(items[col], errors="coerce")
    if not payments.empty:
        for col in ["payment_value", "payment_installments"]:
            if col in payments:
                payments[col] = pd.to_numeric(payments[col], errors="coerce")
    if not reviews.empty and "review_score" in reviews:
        reviews["review_score"] = pd.to_numeric(reviews["review_score"], errors="coerce")

    # Translate category labels
    if not products.empty and not translation.empty:
        if {"product_category_name", "product_category_name_english"}.issubset(translation.columns):
            products = products.merge(
                translation[["product_category_name", "product_category_name_english"]].drop_duplicates(
                    "product_category_name"
                ),
                on="product_category_name", how="left"
            )
            products["category"] = products["product_category_name_english"].fillna(
                products.get("product_category_name", pd.Series(index=products.index, dtype="object"))
            )
        elif "product_category_name" in products:
            products["category"] = products["product_category_name"].fillna("Unknown")
    elif not products.empty and "product_category_name" in products:
        products["category"] = products["product_category_name"].fillna("Unknown")

    if not products.empty and "category" not in products:
        products["category"] = "Unknown"

    # One row per order for order-level analysis
    order_cols = [c for c in [
        "order_id", "customer_id", "order_status", "order_purchase_timestamp",
        "order_approved_at", "order_delivered_carrier_date",
        "order_delivered_customer_date", "order_estimated_delivery_date"
    ] if c in orders.columns]
    order_level = orders[order_cols].drop_duplicates("order_id").copy() if "order_id" in orders else pd.DataFrame()

    # Customer details: one row per customer_id
    if not customers.empty and "customer_id" in customers:
        customer_cols = [c for c in [
            "customer_id", "customer_unique_id", "customer_city", "customer_state"
        ] if c in customers.columns]
        customer_dim = customers[customer_cols].drop_duplicates("customer_id")
        if not order_level.empty:
            order_level = order_level.merge(customer_dim, on="customer_id", how="left", validate="many_to_one")
    else:
        customer_dim = pd.DataFrame()

    # Item-level sales table: one row per order item, never join payment rows here
    sales = items.copy()
    if not sales.empty and "order_id" in sales:
        if not products.empty and "product_id" in products:
            pcols = [c for c in ["product_id", "category", "product_category_name",
                                  "product_name_lenght", "product_description_lenght",
                                  "product_photos_qty", "product_weight_g",
                                  "product_length_cm", "product_height_cm", "product_width_cm"]
                     if c in products.columns]
            sales = sales.merge(
                products[pcols].drop_duplicates("product_id"),
                on="product_id", how="left", validate="many_to_one"
            )
        if not sellers.empty and "seller_id" in sellers:
            scols = [c for c in ["seller_id", "seller_city", "seller_state"] if c in sellers.columns]
            sales = sales.merge(
                sellers[scols].drop_duplicates("seller_id"),
                on="seller_id", how="left", validate="many_to_one"
            )
        if not order_level.empty:
            date_cols = [c for c in ["order_id", "order_purchase_timestamp", "order_status",
                                      "customer_id", "customer_unique_id", "customer_city",
                                      "customer_state"] if c in order_level.columns]
            sales = sales.merge(
                order_level[date_cols].drop_duplicates("order_id"),
                on="order_id", how="left", validate="many_to_one"
            )
    if not sales.empty:
        if "price" in sales:
            sales["price"] = pd.to_numeric(sales["price"], errors="coerce").fillna(0)
        if "freight_value" in sales:
            sales["freight_value"] = pd.to_numeric(sales["freight_value"], errors="coerce").fillna(0)
        if "category" not in sales:
            sales["category"] = "Unknown"
        sales["category"] = sales["category"].fillna("Unknown")

    # Payment table remains at payment-record level; aggregate separately when needed
    payment_data = payments.copy()
    if not payment_data.empty and not order_level.empty and "order_id" in payment_data:
        payment_data = payment_data.merge(
            order_level[[c for c in ["order_id", "order_purchase_timestamp",
                                     "customer_state", "order_status"] if c in order_level.columns]],
            on="order_id", how="left", validate="many_to_one"
        )

    # Review table may contain multiple reviews per order; keep separate from item-level sales
    review_data = reviews.copy()
    if not review_data.empty and not order_level.empty and "order_id" in review_data:
        review_data = review_data.merge(
            order_level[[c for c in ["order_id", "order_purchase_timestamp",
                                     "customer_state", "order_status"] if c in order_level.columns]],
            on="order_id", how="left", validate="many_to_one"
        )

    # Order-level revenue is derived by aggregating item values first
    if not sales.empty and {"order_id", "price"}.issubset(sales.columns):
        order_revenue = sales.groupby("order_id", as_index=False).agg(
            item_revenue=("price", "sum"),
            item_count=("order_id", "size"),
        )
        if "freight_value" in sales:
            freight = sales.groupby("order_id", as_index=False)["freight_value"].sum()
            order_revenue = order_revenue.merge(freight, on="order_id", how="left")
        order_level = order_level.merge(order_revenue, on="order_id", how="left", validate="one_to_one")
    else:
        order_level["item_revenue"] = 0.0
        order_level["item_count"] = 0

    # Payment values are aggregated to one row per order before joining
    if not payment_data.empty and {"order_id", "payment_value"}.issubset(payment_data.columns):
        payment_by_order = payment_data.groupby("order_id", as_index=False).agg(
            payment_total=("payment_value", "sum"),
            payment_record_count=("payment_value", "size"),
        )
        order_level = order_level.merge(payment_by_order, on="order_id", how="left", validate="one_to_one")

    # Delivery measures
    if not order_level.empty:
        if {"order_delivered_customer_date", "order_purchase_timestamp"}.issubset(order_level.columns):
            order_level["delivery_days"] = (
                order_level["order_delivered_customer_date"] - order_level["order_purchase_timestamp"]
            ).dt.total_seconds() / 86400
        if {"order_delivered_customer_date", "order_estimated_delivery_date"}.issubset(order_level.columns):
            order_level["delivery_delay_days"] = (
                order_level["order_delivered_customer_date"] - order_level["order_estimated_delivery_date"]
            ).dt.total_seconds() / 86400
            order_level["is_late"] = order_level["delivery_delay_days"] > 0
        if "order_purchase_timestamp" in order_level:
            order_level["purchase_month"] = order_level["order_purchase_timestamp"].dt.to_period("M").astype(str)

    return {
        "raw": raw, "missing": missing, "orders": order_level, "sales": sales,
        "payments": payment_data, "reviews": review_data, "products": products,
        "customers": customer_dim, "sellers": sellers
    }


# --------------------------- Helpers ---------------------------

def apply_order_filters(df, start_date, end_date, states, statuses):
    if df.empty:
        return df.copy()
    out = df.copy()
    if "order_purchase_timestamp" in out:
        dates = out["order_purchase_timestamp"]
        out = out[dates.notna() & (dates.dt.date >= start_date) & (dates.dt.date <= end_date)]
    if states and "customer_state" in out:
        out = out[out["customer_state"].isin(states)]
    if statuses and "order_status" in out:
        out = out[out["order_status"].isin(statuses)]
    return out


def period_label(start, end):
    return f"{start.strftime('%d %b %Y')} – {end.strftime('%d %b %Y')}"


def kpi_delta(current, previous):
    if previous is None or pd.isna(previous) or previous == 0:
        return None
    return f"{((current - previous) / abs(previous)) * 100:+.1f}% vs previous period"


def make_period_comparison(orders, start, end, states, statuses):
    days = (end - start).days + 1
    prev_end = pd.Timestamp(start) - pd.Timedelta(days=1)
    prev_start = prev_end - pd.Timedelta(days=days - 1)
    previous = apply_order_filters(
        orders, prev_start.date(), prev_end.date(), states, statuses
    )
    current = apply_order_filters(orders, start, end, states, statuses)

    def metrics(frame):
        delivered = frame[frame["order_status"].eq("delivered")] if "order_status" in frame else frame
        rev = frame["item_revenue"].fillna(0).sum() if "item_revenue" in frame else 0
        return {
            "Revenue": rev,
            "Orders": frame["order_id"].nunique() if "order_id" in frame else len(frame),
            "Average order value": safe_divide(rev, frame["order_id"].nunique() if "order_id" in frame else len(frame)),
            "Delivered orders": delivered["order_id"].nunique() if "order_id" in delivered else len(delivered),
        }
    return metrics(current), metrics(previous), prev_start.date(), prev_end.date()


def chart_empty(title, message="No data available for the selected filters."):
    st.subheader(title)
    st.info(message)


# --------------------------- Load and validate ---------------------------

st.title("📊 Olist Insights")
st.caption("Interactive E-Commerce Business Intelligence Dashboard")

data = prepare_data(str(DATA_DIR))
missing_files = data["missing"]
orders_all = data["orders"]
sales_all = data["sales"]
payments_all = data["payments"]
reviews_all = data["reviews"]

if orders_all.empty:
    st.error(
        "The orders file could not be loaded. Place the Olist CSV files in the same folder as app.py. "
        "The expected filename is olist_orders_dataset.csv."
    )
    st.stop()

if missing_files:
    with st.expander("Dataset file check", expanded=True):
        st.warning("Some optional files were not found. Related dashboard sections may be limited.")
        st.write("Missing files:")
        for f in missing_files:
            st.write(f"- {f}")

if "order_purchase_timestamp" not in orders_all or orders_all["order_purchase_timestamp"].notna().sum() == 0:
    st.error("No valid order purchase timestamps were found. Please check the orders CSV.")
    st.stop()

min_date = orders_all["order_purchase_timestamp"].min().date()
max_date = orders_all["order_purchase_timestamp"].max().date()

# --------------------------- Sidebar controls ---------------------------

st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Executive Overview", "Sales Analytics", "Product Analytics",
        "Customer Analytics", "Delivery & Operations", "Payment Analytics",
        "Reviews & Satisfaction", "Business Insights"
    ],
)

st.sidebar.divider()
st.sidebar.subheader("Global Filters")
date_range = st.sidebar.date_input(
    "Order purchase date",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
elif isinstance(date_range, (tuple, list)) and len(date_range) == 1:
    start_date = end_date = date_range[0]
else:
    start_date = end_date = date_range

state_options = sorted(orders_all["customer_state"].dropna().unique().tolist()) if "customer_state" in orders_all else []
status_options = sorted(orders_all["order_status"].dropna().unique().tolist()) if "order_status" in orders_all else []
category_options = sorted(sales_all["category"].dropna().unique().tolist()) if "category" in sales_all else []

selected_states = st.sidebar.multiselect("Customer state", state_options, default=[])
selected_statuses = st.sidebar.multiselect("Order status", status_options, default=[])

if st.sidebar.button("↺ Reset filters", use_container_width=True):
    st.session_state.clear()
    st.rerun()

st.sidebar.divider()
st.sidebar.caption(f"Data period: {min_date:%d %b %Y} – {max_date:%d %b %Y}")
st.sidebar.caption("Filters apply to the relevant analyses. Module-specific filters are shown on each page.")

# Filter order-level data
orders = apply_order_filters(orders_all, start_date, end_date, selected_states, selected_statuses)
selected_order_ids = set(orders["order_id"].dropna().unique()) if "order_id" in orders else set()

# Filter item-level sales through selected order IDs
sales = sales_all[sales_all["order_id"].isin(selected_order_ids)].copy() if not sales_all.empty else sales_all.copy()
# Filter payment and review data separately to avoid join multiplication
payments = payments_all[payments_all["order_id"].isin(selected_order_ids)].copy() if not payments_all.empty else payments_all.copy()
reviews = reviews_all[reviews_all["order_id"].isin(selected_order_ids)].copy() if not reviews_all.empty else reviews_all.copy()

# --------------------------- Common metrics ---------------------------

total_revenue = orders["item_revenue"].fillna(0).sum() if "item_revenue" in orders else 0
total_orders = orders["order_id"].nunique() if "order_id" in orders else len(orders)
unique_customers = orders["customer_unique_id"].nunique() if "customer_unique_id" in orders else 0
aov = safe_divide(total_revenue, total_orders)
delivered = orders[orders["order_status"].eq("delivered")].copy() if "order_status" in orders else orders.iloc[0:0]
avg_review = reviews["review_score"].mean() if not reviews.empty and "review_score" in reviews else np.nan
eligible_delivery = delivered[
    delivered.get("order_delivered_customer_date", pd.Series(index=delivered.index, dtype="datetime64[ns]")).notna()
    & delivered.get("order_estimated_delivery_date", pd.Series(index=delivered.index, dtype="datetime64[ns]")).notna()
] if not delivered.empty else delivered
on_time_rate = (
    (eligible_delivery["order_delivered_customer_date"] <= eligible_delivery["order_estimated_delivery_date"]).mean() * 100
    if not eligible_delivery.empty else np.nan
)

# --------------------------- Page 1: Overview ---------------------------

if page == "Executive Overview":
    st.header("Executive Overview")
    st.markdown('<p class="section-note">A high-level view of sales, customers, orders, and operational performance.</p>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Item Revenue", money(total_revenue))
    c2.metric("Orders", f"{total_orders:,}")
    c3.metric("Average Order Value", money(aov))
    c4.metric("Unique Customers", f"{unique_customers:,}")
    c5.metric("Average Review", f"{avg_review:.2f}/5" if pd.notna(avg_review) else "N/A")

    c6, c7, c8 = st.columns(3)
    c6.metric("Delivered Orders", f"{delivered['order_id'].nunique():,}" if "order_id" in delivered else f"{len(delivered):,}")
    c7.metric("On-Time Delivery", pct(on_time_rate) if pd.notna(on_time_rate) else "N/A")
    c8.metric("Selected Period", period_label(start_date, end_date))

    st.divider()
    left, right = st.columns([1.6, 1])
    with left:
        st.subheader("Revenue Trend")
        granularity = st.selectbox("Time grouping", ["Monthly", "Quarterly", "Yearly"], key="overview_granularity")
        if not orders.empty:
            trend = orders.dropna(subset=["order_purchase_timestamp"]).copy()
            if granularity == "Monthly":
                trend["period"] = trend["order_purchase_timestamp"].dt.to_period("M").astype(str)
            elif granularity == "Quarterly":
                trend["period"] = trend["order_purchase_timestamp"].dt.to_period("Q").astype(str)
            else:
                trend["period"] = trend["order_purchase_timestamp"].dt.year.astype("Int64").astype(str)
            trend = trend.groupby("period", as_index=False)["item_revenue"].sum()
            fig = px.line(trend, x="period", y="item_revenue", markers=True, title="Item revenue over time")
            fig.update_yaxes(title="Revenue (BRL)")
            st.plotly_chart(style_chart(fig), use_container_width=True)
        else:
            chart_empty("Revenue Trend")
    with right:
        st.subheader("Order Status")
        if not orders.empty and "order_status" in orders:
            status = orders["order_status"].value_counts().rename_axis("status").reset_index(name="orders")
            fig = px.pie(status, names="status", values="orders", hole=.48)
            st.plotly_chart(style_chart(fig, 360), use_container_width=True)
        else:
            chart_empty("Order Status")

    left, right = st.columns(2)
    with left:
        st.subheader("Revenue by Product Category")
        if not sales.empty:
            cat = sales.groupby("category", as_index=False)["price"].sum().sort_values("price", ascending=False).head(10)
            fig = px.bar(cat.sort_values("price"), x="price", y="category", orientation="h",
                         title="Top 10 categories by item revenue", labels={"price": "Revenue (BRL)", "category": "Category"})
            st.plotly_chart(style_chart(fig), use_container_width=True)
        else:
            chart_empty("Revenue by Product Category")
    with right:
        st.subheader("Orders by Customer State")
        if not orders.empty and "customer_state" in orders:
            geo = orders.groupby("customer_state", as_index=False)["order_id"].nunique().sort_values("order_id", ascending=False).head(10)
            fig = px.bar(geo, x="customer_state", y="order_id", title="Top states by order count",
                         labels={"customer_state": "State", "order_id": "Orders"})
            st.plotly_chart(style_chart(fig), use_container_width=True)
        else:
            chart_empty("Orders by Customer State")

    st.subheader("Period-over-Period Comparison")
    current, previous, prev_start, prev_end = make_period_comparison(
        orders_all, start_date, end_date, selected_states, selected_statuses
    )
    st.caption(f"Current: {period_label(start_date, end_date)}  |  Previous: {period_label(prev_start, prev_end)}")
    comp = pd.DataFrame({
        "Metric": current.keys(),
        "Selected period": [current[k] for k in current],
        "Previous period": [previous[k] for k in current],
        "Change": [kpi_delta(current[k], previous[k]) or "N/A" for k in current],
    })
    st.dataframe(comp, use_container_width=True, hide_index=True)

# --------------------------- Page 2: Sales ---------------------------

elif page == "Sales Analytics":
    st.header("Sales Analytics")
    st.caption("Explore revenue, order volume, average order value, and sales trends.")

    metric_choice = st.selectbox("Analyse by", ["Revenue", "Order count", "Average order value", "Average item price"])
    grouping = st.selectbox("Time grouping", ["Monthly", "Quarterly", "Yearly"], key="sales_grouping")
    if not sales.empty and not orders.empty:
        sales_metric = sales.copy()
        sales_metric = sales_metric.dropna(subset=["order_purchase_timestamp"])
        if grouping == "Monthly":
            sales_metric["period"] = sales_metric["order_purchase_timestamp"].dt.to_period("M").astype(str)
        elif grouping == "Quarterly":
            sales_metric["period"] = sales_metric["order_purchase_timestamp"].dt.to_period("Q").astype(str)
        else:
            sales_metric["period"] = sales_metric["order_purchase_timestamp"].dt.year.astype("Int64").astype(str)

        if metric_choice == "Revenue":
            metric_df = sales_metric.groupby("period", as_index=False)["price"].sum().rename(columns={"price": "value"})
        elif metric_choice == "Order count":
            metric_df = sales_metric.groupby("period")["order_id"].nunique().reset_index(name="value")
        elif metric_choice == "Average order value":
            rev = sales_metric.groupby("period")["price"].sum()
            count = sales_metric.groupby("period")["order_id"].nunique()
            metric_df = (rev / count.replace(0, np.nan)).reset_index(name="value").fillna(0)
        else:
            metric_df = sales_metric.groupby("period", as_index=False)["price"].mean().rename(columns={"price": "value"})
        fig = px.line(metric_df, x="period", y="value", markers=True, title=f"{metric_choice} over time")
        st.plotly_chart(style_chart(fig, 400), use_container_width=True)
    else:
        empty_message("Sales data is not available for the selected filters.")

    left, right = st.columns(2)
    with left:
        st.subheader("Category Performance")
        top_n = st.slider("Number of categories", 5, 20, 10, key="sales_topn")
        if not sales.empty:
            cat = sales.groupby("category", as_index=False).agg(
                revenue=("price", "sum"), items=("order_id", "size"), orders=("order_id", "nunique")
            ).sort_values("revenue", ascending=False).head(top_n)
            fig = px.bar(cat.sort_values("revenue"), x="revenue", y="category", orientation="h",
                         title=f"Top {top_n} categories by revenue")
            st.plotly_chart(style_chart(fig), use_container_width=True)
        else:
            empty_message("No sales records for the selected filters.")
    with right:
        st.subheader("State-wise Sales")
        if not sales.empty and "customer_state" in sales:
            state_df = sales.groupby("customer_state", as_index=False).agg(
                revenue=("price", "sum"), orders=("order_id", "nunique")
            ).sort_values("revenue", ascending=False).head(15)
            fig = px.bar(state_df, x="customer_state", y="revenue", title="Revenue by customer state")
            st.plotly_chart(style_chart(fig), use_container_width=True)
        else:
            empty_message("State information is not available.")

    st.subheader("Sales Detail")
    if not sales.empty:
        detail_cols = [c for c in ["order_id", "order_purchase_timestamp", "product_id", "category",
                                    "seller_id", "customer_state", "price", "freight_value"] if c in sales]
        detail = sales[detail_cols].copy()
        st.dataframe(detail.head(1000), use_container_width=True, hide_index=True)
        download_csv(detail, "olist_filtered_sales.csv", "⬇ Download filtered sales")
        st.caption("The table preview is limited to the first 1,000 rows; the CSV export includes all filtered rows.")

# --------------------------- Page 3: Products ---------------------------

elif page == "Product Analytics":
    st.header("Product Analytics")
    st.caption("Compare categories and products using item-level sales data.")

    if sales.empty:
        empty_message("No product sales records are available for the selected filters.")
    else:
        cats = sorted(sales["category"].dropna().unique().tolist())
        chosen_cats = st.multiselect("Product categories", cats, default=cats[:min(5, len(cats))])
        product_view = sales[sales["category"].isin(chosen_cats)].copy() if chosen_cats else sales.iloc[0:0].copy()
        top_n = st.slider("Show top products", 5, 30, 10)
        product_metric = st.selectbox("Rank products by", ["Revenue", "Quantity of order items", "Average item price"])
        if not product_view.empty:
            if product_metric == "Revenue":
                ranking = product_view.groupby("product_id", as_index=False)["price"].sum().rename(columns={"price": "metric"})
            elif product_metric == "Quantity of order items":
                ranking = product_view.groupby("product_id", as_index=False).size().rename(columns={"size": "metric"})
            else:
                ranking = product_view.groupby("product_id", as_index=False)["price"].mean().rename(columns={"price": "metric"})
            ranking = ranking.sort_values("metric", ascending=False).head(top_n)
            fig = px.bar(ranking.sort_values("metric"), x="metric", y="product_id", orientation="h",
                         title=f"Top {top_n} products by {product_metric.lower()}")
            st.plotly_chart(style_chart(fig, 450), use_container_width=True)

            category_summary = product_view.groupby("category", as_index=False).agg(
                revenue=("price", "sum"),
                order_items=("order_id", "size"),
                distinct_orders=("order_id", "nunique"),
                average_item_price=("price", "mean"),
            ).sort_values("revenue", ascending=False)
            st.subheader("Category Comparison")
            st.dataframe(category_summary, use_container_width=True, hide_index=True)
            download_csv(category_summary, "olist_category_comparison.csv")

            st.subheader("Product Detail Explorer")
            search_id = st.text_input("Search by product ID (optional)")
            detail = product_view.copy()
            if search_id.strip() and "product_id" in detail:
                detail = detail[detail["product_id"].astype(str).str.contains(search_id.strip(), case=False, na=False)]
            detail_cols = [c for c in ["product_id", "category", "order_id", "price", "freight_value",
                                        "seller_id", "customer_state"] if c in detail]
            st.dataframe(detail[detail_cols].head(1000), use_container_width=True, hide_index=True)
            download_csv(detail[detail_cols], "olist_product_details.csv")
        else:
            empty_message("Choose at least one category to display product analytics.")

# --------------------------- Page 4: Customers ---------------------------

elif page == "Customer Analytics":
    st.header("Customer Analytics")
    st.caption("Explore customer purchasing behaviour and RFM-based customer segments.")

    if orders.empty or "customer_unique_id" not in orders:
        empty_message("Customer identifiers are not available in the current data.")
    else:
        customer_orders = orders.dropna(subset=["customer_unique_id"]).copy()
        customer_orders = customer_orders[customer_orders["order_status"].eq("delivered")] if "order_status" in customer_orders else customer_orders
        reference_date = pd.Timestamp(end_date) + pd.Timedelta(days=1)
        rfm = customer_orders.groupby("customer_unique_id").agg(
            last_purchase=("order_purchase_timestamp", "max"),
            frequency=("order_id", "nunique"),
            monetary=("item_revenue", "sum"),
        ).reset_index()
        rfm["recency_days"] = (reference_date - rfm["last_purchase"]).dt.days
        rfm = rfm.replace([np.inf, -np.inf], np.nan).dropna(subset=["recency_days", "frequency", "monetary"])

        if not rfm.empty:
            # Percentile-rank scoring avoids qcut failures when the filtered
            # population is small or many customers have identical values.
            rfm["R_score"] = (5 - np.ceil(rfm["recency_days"].rank(pct=True) * 4)).clip(1, 4).astype(int)
            rfm["F_score"] = np.ceil(rfm["frequency"].rank(pct=True) * 4).clip(1, 4).astype(int)
            rfm["M_score"] = np.ceil(rfm["monetary"].rank(pct=True) * 4).clip(1, 4).astype(int)
            rfm["RFM_score"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]

            def segment(score):
                if score >= 10:
                    return "High value"
                if score >= 7:
                    return "Loyal / promising"
                if score >= 5:
                    return "Needs attention"
                return "At risk / inactive"

            rfm["segment"] = rfm["RFM_score"].apply(segment)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Customers analysed", f"{len(rfm):,}")
            c2.metric("Repeat customers", f"{(rfm['frequency'] > 1).sum():,}")
            c3.metric("Repeat customer rate", pct((rfm["frequency"] > 1).mean() * 100))
            c4.metric("Average spend / customer", money(rfm["monetary"].mean()))

            left, right = st.columns(2)
            with left:
                seg = rfm["segment"].value_counts().rename_axis("segment").reset_index(name="customers")
                fig = px.pie(seg, names="segment", values="customers", hole=.45, title="Customer segments")
                st.plotly_chart(style_chart(fig), use_container_width=True)
            with right:
                fig = px.scatter(rfm, x="frequency", y="monetary", color="segment",
                                 hover_data=["recency_days"], title="Customer frequency vs spending",
                                 labels={"frequency": "Distinct orders", "monetary": "Item revenue (BRL)"})
                st.plotly_chart(style_chart(fig), use_container_width=True)

            st.subheader("Customer Segment Explorer")
            chosen_segments = st.multiselect("Segments", sorted(rfm["segment"].unique()), default=sorted(rfm["segment"].unique()))
            min_orders = st.number_input("Minimum distinct orders", min_value=1, max_value=int(max(1, rfm["frequency"].max())), value=1)
            min_spend = st.number_input("Minimum item spending (BRL)", min_value=0.0, value=0.0, step=50.0)
            customer_view = rfm[
                rfm["segment"].isin(chosen_segments)
                & (rfm["frequency"] >= min_orders)
                & (rfm["monetary"] >= min_spend)
            ].sort_values("monetary", ascending=False)
            st.dataframe(customer_view.head(1000), use_container_width=True, hide_index=True)
            download_csv(customer_view, "olist_customer_rfm_segments.csv")
            st.caption("RFM scores are relative to the currently filtered, delivered-customer population. Segment labels are analytical groupings, not official Olist classifications.")
        else:
            empty_message("Not enough valid customer purchase data to calculate RFM segments.")

# --------------------------- Page 5: Delivery ---------------------------

elif page == "Delivery & Operations":
    st.header("Delivery & Operations")
    st.caption("Investigate delivery duration, late deliveries, and operational patterns.")

    if delivered.empty or "delivery_days" not in delivered:
        empty_message("Delivery timestamp data is not available for the selected filters.")
    else:
        valid = delivered[delivered["delivery_days"].notna() & (delivered["delivery_days"] >= 0)].copy()
        valid_est = valid.dropna(subset=["order_delivered_customer_date", "order_estimated_delivery_date"]).copy()
        if not valid.empty:
            late_rate = valid_est["is_late"].mean() * 100 if "is_late" in valid_est and not valid_est.empty else np.nan
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Eligible delivered orders", f"{len(valid):,}")
            c2.metric("Average delivery days", f"{valid['delivery_days'].mean():.1f}")
            c3.metric("Median delivery days", f"{valid['delivery_days'].median():.1f}")
            c4.metric("Late delivery rate", pct(late_rate) if pd.notna(late_rate) else "N/A")

            left, right = st.columns(2)
            with left:
                fig = px.histogram(valid, x="delivery_days", nbins=30, title="Distribution of delivery duration",
                                   labels={"delivery_days": "Delivery days"})
                st.plotly_chart(style_chart(fig), use_container_width=True)
            with right:
                if not valid_est.empty:
                    late = valid_est["is_late"].value_counts().rename_axis("late").reset_index(name="orders")
                    late["Delivery result"] = late["late"].map({True: "Late", False: "On time"})
                    fig = px.pie(late, names="Delivery result", values="orders", hole=.45,
                                 title="On-time vs late deliveries")
                    st.plotly_chart(style_chart(fig), use_container_width=True)
                else:
                    empty_message("Estimated delivery timestamps are unavailable.")

            if "customer_state" in valid:
                state_delivery = valid.groupby("customer_state", as_index=False).agg(
                    avg_delivery_days=("delivery_days", "mean"),
                    delivered_orders=("order_id", "nunique")
                )
                if not valid_est.empty:
                    state_late = valid_est.groupby("customer_state", as_index=False)["is_late"].mean()
                    state_late["late_rate_pct"] = state_late["is_late"] * 100
                    state_delivery = state_delivery.merge(state_late[["customer_state", "late_rate_pct"]], on="customer_state", how="left")
                st.subheader("Delivery Performance by State")
                fig = px.bar(state_delivery.sort_values("avg_delivery_days", ascending=False),
                             x="customer_state", y="avg_delivery_days", title="Average delivery days by state")
                st.plotly_chart(style_chart(fig), use_container_width=True)

            if not reviews.empty and "review_score" in reviews:
                review_order = reviews.groupby("order_id", as_index=False)["review_score"].mean()
                delivery_review = valid[["order_id", "delivery_days"]].merge(review_order, on="order_id", how="inner")
                if not delivery_review.empty:
                    st.subheader("Delivery Duration vs Review Score")
                    fig = px.scatter(delivery_review, x="delivery_days", y="review_score",
                                     opacity=.45, title="Explore the relationship between delivery time and review score")
                    st.plotly_chart(style_chart(fig), use_container_width=True)
                    st.caption("This chart shows association, not proof that delivery duration causes a particular review score.")

            st.subheader("Delivery Records")
            delivery_cols = [c for c in ["order_id", "customer_state", "order_purchase_timestamp",
                                         "order_delivered_customer_date", "order_estimated_delivery_date",
                                         "delivery_days", "delivery_delay_days", "is_late"] if c in valid]
            delivery_detail = valid[delivery_cols].copy()
            late_only = st.checkbox("Show only late deliveries")
            if late_only and "is_late" in delivery_detail:
                delivery_detail = delivery_detail[delivery_detail["is_late"] == True]
            st.dataframe(delivery_detail.head(1000), use_container_width=True, hide_index=True)
            download_csv(delivery_detail, "olist_delivery_records.csv")
        else:
            empty_message("No valid delivery durations are available.")

# --------------------------- Page 6: Payments ---------------------------

elif page == "Payment Analytics":
    st.header("Payment Analytics")
    st.caption("Explore payment methods, payment values, and installment patterns.")

    if payments.empty or "payment_value" not in payments:
        empty_message("Payment data is not available for the selected filters.")
    else:
        payments["payment_value"] = pd.to_numeric(payments["payment_value"], errors="coerce")
        pay_total = payments["payment_value"].sum()
        payment_orders = payments["order_id"].nunique() if "order_id" in payments else 0
        c1, c2, c3 = st.columns(3)
        c1.metric("Payment value", money(pay_total))
        c2.metric("Orders with payment records", f"{payment_orders:,}")
        c3.metric("Average payment record", money(payments["payment_value"].mean()))

        left, right = st.columns(2)
        with left:
            if "payment_type" in payments:
                by_type = payments.groupby("payment_type", as_index=False)["payment_value"].sum().sort_values("payment_value", ascending=False)
                fig = px.bar(by_type, x="payment_type", y="payment_value", title="Payment value by method")
                st.plotly_chart(style_chart(fig), use_container_width=True)
        with right:
            if "payment_installments" in payments:
                inst = payments.groupby("payment_installments", as_index=False).size().rename(columns={"size": "records"})
                fig = px.bar(inst, x="payment_installments", y="records", title="Payment records by installment count")
                st.plotly_chart(style_chart(fig), use_container_width=True)

        if "payment_type" in payments:
            method_options = sorted(payments["payment_type"].dropna().unique().tolist())
            selected_methods = st.multiselect("Payment methods", method_options, default=method_options)
            payment_view = payments[payments["payment_type"].isin(selected_methods)].copy()
        else:
            payment_view = payments.copy()
        st.subheader("Payment Detail")
        st.dataframe(payment_view.head(1000), use_container_width=True, hide_index=True)
        download_csv(payment_view, "olist_payment_records.csv")
        st.caption("Payment value is aggregated from payment records. One order may contain multiple payment records; payment-record count is not the same as order count.")

# --------------------------- Page 7: Reviews ---------------------------

elif page == "Reviews & Satisfaction":
    st.header("Reviews & Customer Satisfaction")
    st.caption("Explore review scores and investigate their relationship with products and delivery.")

    if reviews.empty or "review_score" not in reviews:
        empty_message("Review data is not available for the selected filters.")
    else:
        reviews["review_score"] = pd.to_numeric(reviews["review_score"], errors="coerce")
        review_valid = reviews.dropna(subset=["review_score"]).copy()
        c1, c2, c3 = st.columns(3)
        c1.metric("Reviews", f"{len(review_valid):,}")
        c2.metric("Average review score", f"{review_valid['review_score'].mean():.2f}/5" if not review_valid.empty else "N/A")
        c3.metric("Low scores (1–2)", pct(review_valid["review_score"].isin([1, 2]).mean() * 100) if not review_valid.empty else "N/A")

        left, right = st.columns(2)
        with left:
            score_counts = review_valid["review_score"].value_counts().sort_index().rename_axis("score").reset_index(name="reviews")
            fig = px.bar(score_counts, x="score", y="reviews", title="Review score distribution")
            st.plotly_chart(style_chart(fig), use_container_width=True)
        with right:
            if "order_purchase_timestamp" in review_valid:
                trend = review_valid.dropna(subset=["order_purchase_timestamp"]).copy()
                trend["month"] = trend["order_purchase_timestamp"].dt.to_period("M").astype(str)
                trend = trend.groupby("month", as_index=False)["review_score"].mean()
                fig = px.line(trend, x="month", y="review_score", markers=True, title="Average review score over time")
                st.plotly_chart(style_chart(fig), use_container_width=True)

        if not sales.empty and "category" in sales:
            category_reviews = sales[["order_id", "category"]].drop_duplicates().merge(
                review_valid.groupby("order_id", as_index=False)["review_score"].mean(),
                on="order_id", how="inner"
            )
            category_reviews = category_reviews.groupby("category", as_index=False)["review_score"].mean().sort_values("review_score")
            st.subheader("Average Review Score by Product Category")
            fig = px.bar(category_reviews.head(15), x="review_score", y="category", orientation="h")
            st.plotly_chart(style_chart(fig), use_container_width=True)

        score_filter = st.multiselect("Filter review scores", [1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])
        review_view = review_valid[review_valid["review_score"].isin(score_filter)]
        st.subheader("Review Detail")
        st.dataframe(review_view.head(1000), use_container_width=True, hide_index=True)
        download_csv(review_view, "olist_review_records.csv")

# --------------------------- Page 8: Business Insights ---------------------------

elif page == "Business Insights":
    st.header("Business Insights & Recommendations")
    st.caption("Evidence-based observations generated from the currently selected filters.")

    insights = []
    if not orders.empty:
        if total_revenue > 0:
            insights.append(("Revenue", f"Item revenue for the selected period is {money(total_revenue)} across {total_orders:,} distinct orders.",
                             "Use the sales and category pages to identify the products and locations contributing most to this result."))
        if "order_status" in orders:
            delivered_share = safe_divide(orders["order_status"].eq("delivered").sum(), len(orders)) * 100
            insights.append(("Order fulfilment", f"{delivered_share:.1f}% of selected order records have delivered status.",
                             "Review the order-status distribution and investigate cancelled, unavailable, or still-processing orders separately."))
        if not sales.empty:
            cat = sales.groupby("category")["price"].sum().sort_values(ascending=False)
            if len(cat):
                share = safe_divide(cat.iloc[0], sales["price"].sum()) * 100
                insights.append(("Category concentration", f"The leading category is '{cat.index[0]}' and contributes {share:.1f}% of selected item revenue.",
                                 "Compare category concentration over different periods before changing inventory or marketing priorities."))
        if pd.notna(on_time_rate):
            insights.append(("Delivery", f"{on_time_rate:.1f}% of eligible delivered orders arrived on or before the estimated delivery date.",
                             "Use the delivery page to identify states and periods with higher late-delivery rates."))
        if pd.notna(avg_review):
            insights.append(("Customer satisfaction", f"The average available review score is {avg_review:.2f} out of 5.",
                             "Investigate low-score reviews alongside delivery and product-category patterns; these relationships do not establish causation."))

    if insights:
        for i, (heading, finding, action) in enumerate(insights, start=1):
            with st.container(border=True):
                st.markdown(f"**{i}. {heading}**")
                st.write(finding)
                st.markdown(f"**Suggested investigation:** {action}")
    else:
        empty_message("There is not enough data under the current filters to generate insights.")

    st.divider()
    st.subheader("Build a custom summary")
    summary_metric = st.selectbox("Choose a metric", ["Revenue", "Orders", "Average order value", "Unique customers"])
    if summary_metric == "Revenue":
        st.metric("Selected metric", money(total_revenue))
    elif summary_metric == "Orders":
        st.metric("Selected metric", f"{total_orders:,}")
    elif summary_metric == "Average order value":
        st.metric("Selected metric", money(aov))
    else:
        st.metric("Selected metric", f"{unique_customers:,}")

    summary = pd.DataFrame({
        "Metric": ["Item revenue (BRL)", "Distinct orders", "Average order value (BRL)",
                   "Unique customers", "Average review score", "On-time delivery rate (%)"],
        "Value": [total_revenue, total_orders, aov, unique_customers,
                  avg_review if pd.notna(avg_review) else np.nan,
                  on_time_rate if pd.notna(on_time_rate) else np.nan],
    })
    download_csv(summary, "olist_business_summary.csv", "⬇ Download business summary")


# --------------------------- Footer / Data notes ---------------------------

st.divider()
with st.expander("About the data and metric definitions"):
    st.markdown("""
    - **Item revenue:** Sum of `price` from order-item records. Freight and payment values are not included.
    - **Orders:** Distinct order IDs after the selected order-level filters.
    - **Average order value:** Item revenue divided by distinct selected orders. It is a dataset-level measure, not necessarily accounting revenue.
    - **Unique customers:** Distinct `customer_unique_id` values in the selected order population.
    - **Delivery duration:** Delivered-customer timestamp minus purchase timestamp, in days.
    - **Late delivery:** Actual delivery timestamp is later than the estimated delivery timestamp. Orders missing either timestamp are excluded from the eligible on-time/late calculation.
    - **Payment value:** Sum of payment records. One order can have multiple payment records, so payment rows must not be counted as orders.
    - **RFM segmentation:** Relative quartile-based groups calculated from the currently selected delivered-customer population.

    The Olist dataset is historical (2016–2018). Dashboard results describe the dataset and should not be interpreted as current Brazilian e-commerce market conditions. Correlation does not establish causation.
    """)
st.caption("Olist Insights • Built with Python, Streamlit, Pandas and Plotly")
