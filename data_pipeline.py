from pathlib import Path
import json
import pandas as pd
import numpy as np

DATA_DIR = Path("data")


def read_csv(filename):
    path = DATA_DIR / filename

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def read_json(filename):
    path = DATA_DIR / filename

    if not path.exists():
        return pd.DataFrame()

    try:
        with open(path, "r", encoding="utf-8") as file:
            return pd.DataFrame(json.load(file))
    except json.JSONDecodeError:
        return pd.DataFrame()


def read_log(filename):
    path = DATA_DIR / filename

    if not path.exists():
        return pd.DataFrame(columns=["timestamp", "event"])

    rows = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if " | " in line:
                timestamp, event = line.strip().split(" | ", 1)
                rows.append({"timestamp": timestamp, "event": event})

    return pd.DataFrame(rows)


def ingest_data():
    return {
        "sales": read_csv("sales_data.csv"),
        "marketing": read_csv("marketing_data.csv"),
        "advertising": read_csv("advertising_data.csv"),
        "hr": read_csv("hr_data.csv"),
        "customers": read_csv("customer_data.csv"),
        "web": read_json("web_logs.json"),
        "logs": read_log("system_activity.log"),
    }


def clean_data(raw):
    # IMPORTANT: cleaning happens before validation.
    cleaned = {name: df.copy() for name, df in raw.items()}

    for name, df in cleaned.items():
        if not df.empty:
            df.drop_duplicates(inplace=True)

    date_columns = {
        "sales": ["date"],
        "marketing": ["date"],
        "advertising": ["date"],
        "hr": ["date"],
        "customers": ["date"],
        "web": ["timestamp"],
        "logs": ["timestamp"],
    }

    numeric_columns = {
        "sales": ["revenue", "cost", "profit", "sales_cycle_days", "latitude", "longitude"],
        "marketing": ["impressions", "clicks", "leads", "conversions", "campaign_cost", "revenue_generated", "latitude", "longitude"],
        "advertising": ["impressions", "clicks", "conversions", "ad_spend", "attributed_revenue", "latitude", "longitude"],
        "hr": ["training_hours", "attendance_rate", "workload_score", "productivity_score", "performance_score", "revenue_contribution"],
        "customers": ["latitude", "longitude", "lifetime_value"],
        "web": ["latitude", "longitude", "page_views", "bounce_rate", "status_code", "response_time"],
    }

    for dataset, columns in date_columns.items():
        df = cleaned.get(dataset, pd.DataFrame())

        if df.empty:
            continue

        for column in columns:
            if column in df.columns:
                df[column] = pd.to_datetime(df[column], errors="coerce")

    for dataset, columns in numeric_columns.items():
        df = cleaned.get(dataset, pd.DataFrame())

        if df.empty:
            continue

        for column in columns:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    sales = cleaned["sales"]

    if not sales.empty:
        sales["profit"] = sales["revenue"] - sales["cost"]
        sales["profit_margin"] = np.where(sales["revenue"] > 0, sales["profit"] / sales["revenue"] * 100, 0)
        sales["month"] = sales["date"].dt.to_period("M").astype(str)

    marketing = cleaned["marketing"]

    if not marketing.empty:
        marketing["ctr"] = np.where(marketing["impressions"] > 0, marketing["clicks"] / marketing["impressions"] * 100, 0)
        marketing["conversion_rate"] = np.where(marketing["leads"] > 0, marketing["conversions"] / marketing["leads"] * 100, 0)
        marketing["marketing_roi"] = np.where(
            marketing["campaign_cost"] > 0,
            (marketing["revenue_generated"] - marketing["campaign_cost"]) / marketing["campaign_cost"] * 100,
            0,
        )
        marketing["month"] = marketing["date"].dt.to_period("M").astype(str)

    advertising = cleaned["advertising"]

    if not advertising.empty:
        advertising["cpc"] = np.where(advertising["clicks"] > 0, advertising["ad_spend"] / advertising["clicks"], 0)
        advertising["cpa"] = np.where(advertising["conversions"] > 0, advertising["ad_spend"] / advertising["conversions"], 0)
        advertising["ctr"] = np.where(advertising["impressions"] > 0, advertising["clicks"] / advertising["impressions"] * 100, 0)
        advertising["roas"] = np.where(advertising["ad_spend"] > 0, advertising["attributed_revenue"] / advertising["ad_spend"], 0)
        advertising["roi_percent"] = np.where(
            advertising["ad_spend"] > 0,
            (advertising["attributed_revenue"] - advertising["ad_spend"]) / advertising["ad_spend"] * 100,
            0,
        )
        advertising["month"] = advertising["date"].dt.to_period("M").astype(str)

    customers = cleaned["customers"]

    if not customers.empty:
        customers["converted_flag"] = customers["conversion_status"].eq("Converted").astype(int)
        customers["month"] = customers["date"].dt.to_period("M").astype(str)

    web = cleaned["web"]

    if not web.empty:
        web["date"] = web["timestamp"].dt.date
        web["hour"] = web["timestamp"].dt.hour
        web["success"] = web["status_code"].eq(200)

    return cleaned


def validate_data(cleaned):
    # IMPORTANT: validation happens after cleaning.
    required_columns = {
        "sales": ["date", "client_id", "campaign_id", "country", "region", "service", "revenue", "cost", "profit"],
        "marketing": ["date", "campaign_id", "channel", "impressions", "clicks", "leads", "conversions"],
        "advertising": ["date", "campaign_id", "platform", "ad_spend", "conversions", "attributed_revenue"],
        "hr": ["employee_id", "department", "performance_score", "training_hours", "attendance_rate"],
        "customers": ["client_id", "segment", "region", "conversion_status", "lifetime_value"],
        "web": ["timestamp", "client_id", "campaign_id", "region", "service_viewed", "traffic_source"],
    }

    rows = []

    for dataset, columns in required_columns.items():
        df = cleaned.get(dataset, pd.DataFrame())
        missing = [column for column in columns if column not in df.columns]

        rows.append({
            "dataset": dataset,
            "rows": len(df),
            "missing_columns": ", ".join(missing) if missing else "None",
            "valid": len(missing) == 0 and len(df) > 0,
        })

    return pd.DataFrame(rows)


def join_data(cleaned):
    sales = cleaned["sales"]
    marketing = cleaned["marketing"]
    advertising = cleaned["advertising"]
    customers = cleaned["customers"]
    web = cleaned["web"]
    hr = cleaned["hr"]

    joined = {}

    if not sales.empty and not customers.empty:
        joined["sales_customer"] = sales.merge(
            customers[[
                "client_id",
                "industry",
                "lead_source",
                "lifetime_value",
                "conversion_status",
                "repeat_customer",
            ]],
            on="client_id",
            how="left",
        )
    else:
        joined["sales_customer"] = sales

    if not marketing.empty and not advertising.empty:
        joined["marketing_advertising"] = marketing.merge(
            advertising[[
                "campaign_id",
                "platform",
                "ad_spend",
                "attributed_revenue",
                "roas",
                "cpa",
                "cpc",
                "roi_percent",
            ]],
            on="campaign_id",
            how="left",
        )
    else:
        joined["marketing_advertising"] = marketing

    if not web.empty and not customers.empty:
        joined["web_customer"] = web.merge(
            customers[[
                "client_id",
                "segment",
                "industry",
                "conversion_status",
                "lifetime_value",
            ]],
            on="client_id",
            how="left",
        )
    else:
        joined["web_customer"] = web

    if not sales.empty and not marketing.empty:
        joined["sales_marketing"] = sales.merge(
            marketing[[
                "campaign_id",
                "campaign",
                "channel",
                "leads",
                "conversions",
                "campaign_cost",
                "revenue_generated",
            ]],
            on="campaign_id",
            how="left",
        )
    else:
        joined["sales_marketing"] = sales

    joined["hr_sales_context"] = hr.copy()

    return joined


def apply_filters(data, country="All", region="All", period="All", segment="All", service="All"):
    filtered = {}

    for name, df in data.items():
        if df.empty:
            filtered[name] = df
            continue

        temp = df.copy()

        if country != "All" and "country" in temp.columns:
            temp = temp[temp["country"] == country]

        if region != "All" and "region" in temp.columns:
            temp = temp[temp["region"] == region]

        if segment != "All" and "segment" in temp.columns:
            temp = temp[temp["segment"] == segment]

        service_column = None

        for possible_column in ["service", "service_interest", "service_viewed"]:
            if possible_column in temp.columns:
                service_column = possible_column
                break

        if service != "All" and service_column:
            temp = temp[temp[service_column] == service]

        if period != "All":
            date_column = None

            for possible_column in ["date", "timestamp"]:
                if possible_column in temp.columns:
                    date_column = possible_column
                    break

            if date_column:
                temp[date_column] = pd.to_datetime(temp[date_column], errors="coerce")

                if period == "Last 7 days":
                    temp = temp[temp[date_column] >= pd.Timestamp.now() - pd.Timedelta(days=7)]
                elif period == "Last 30 days":
                    temp = temp[temp[date_column] >= pd.Timestamp.now() - pd.Timedelta(days=30)]
                elif period == "Last 90 days":
                    temp = temp[temp[date_column] >= pd.Timestamp.now() - pd.Timedelta(days=90)]
                elif period == "Historic":
                    temp = temp[temp[date_column] < pd.Timestamp.now() - pd.Timedelta(days=30)]

        filtered[name] = temp

    return filtered


def run_pipeline():
    raw = ingest_data()
    cleaned = clean_data(raw)
    validation_report = validate_data(cleaned)
    joined = join_data(cleaned)

    return cleaned, validation_report, joined