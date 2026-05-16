import pandas as pd
import numpy as np


def safe_sum(df, column):
    if df.empty or column not in df.columns:
        return 0
    return df[column].sum()


def safe_mean(df, column):
    if df.empty or column not in df.columns:
        return 0
    return df[column].mean()


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce").fillna(0)


def safe_ratio(numerator, denominator, default=0):
    denominator = float(denominator) if denominator not in [None, 0] else 0
    if denominator == 0:
        return default
    return numerator / denominator


def executive_kpis(data):
    sales = data["sales"]
    marketing = data["marketing"]
    advertising = data["advertising"]
    customers = data["customers"]

    revenue = safe_sum(sales, "revenue")
    sales_cost = safe_sum(sales, "cost")
    marketing_cost = safe_sum(marketing, "campaign_cost")
    ad_cost = safe_sum(advertising, "ad_spend")

    total_cost = sales_cost + marketing_cost + ad_cost
    profit = revenue - total_cost
    roi = (profit / total_cost * 100) if total_cost > 0 else 0

    conversions = safe_sum(marketing, "conversions") + safe_sum(advertising, "conversions")
    forecast_confidence = min(95, 70 + int(len(sales) / 25))

    return {
        "revenue": revenue,
        "profit": profit,
        "roi": roi,
        "conversions": int(conversions),
        "customers": len(customers),
        "forecast_confidence": forecast_confidence,
    }


def region_summary(data):
    sales = data["sales"]
    web = data["web"]
    customers = data["customers"]
    advertising = data["advertising"]

    pieces = []

    if not sales.empty:
        pieces.append(
            sales.groupby(["country", "region", "latitude", "longitude"], as_index=False).agg(
                revenue=("revenue", "sum"),
                profit=("profit", "sum"),
                deals=("sale_id", "count"),
            )
        )

    if not web.empty:
        pieces.append(
            web.groupby(["country", "region", "latitude", "longitude"], as_index=False).agg(
                traffic=("log_id", "count"),
                avg_bounce=("bounce_rate", "mean"),
                avg_response_time=("response_time", "mean"),
            )
        )

    if not customers.empty:
        pieces.append(
            customers.groupby(["country", "region", "latitude", "longitude"], as_index=False).agg(
                customers=("client_id", "nunique"),
                lifetime_value=("lifetime_value", "sum"),
            )
        )

    if not pieces:
        return pd.DataFrame()

    base = pieces[0]

    for part in pieces[1:]:
        base = base.merge(
            part,
            on=["country", "region", "latitude", "longitude"],
            how="outer",
        )

    if not advertising.empty:
        ads = advertising.groupby(["country", "region"], as_index=False).agg(
            ad_spend=("ad_spend", "sum"),
            attributed_revenue=("attributed_revenue", "sum"),
            ad_conversions=("conversions", "sum"),
        )

        ads["regional_roas"] = np.where(
            ads["ad_spend"] > 0,
            ads["attributed_revenue"] / ads["ad_spend"],
            0,
        )

        base = base.merge(ads, on=["country", "region"], how="left")

    for column in [
        "revenue", "profit", "deals", "traffic", "avg_bounce",
        "avg_response_time", "customers", "lifetime_value",
        "ad_spend", "attributed_revenue", "ad_conversions", "regional_roas"
    ]:
        if column in base.columns:
            base[column] = base[column].fillna(0)

    max_revenue = max(base["revenue"].max(), 1) if "revenue" in base.columns else 1

    base["opportunity_score"] = (
        base.get("traffic", 0) * 0.15
        + base.get("customers", 0) * 0.25
        + base.get("regional_roas", 0) * 8
        + (base.get("revenue", 0) / max_revenue) * 60
    )

    base["market_status"] = np.select(
        [
            base["opportunity_score"] >= 75,
            base["opportunity_score"].between(45, 74),
            base["opportunity_score"].between(20, 44),
        ],
        ["High Growth", "Existing", "Potential"],
        default="Underperforming",
    )

    return base.sort_values("opportunity_score", ascending=False)


def select_forecast_dataset(data, department, metric):
    department = str(department).lower()
    metric = str(metric).lower()

    if department in ["sales", "commercial"]:
        if "win" in metric:
            return data["sales"], "__sales_win_rate__", "Win Rate"
        if "deal" in metric:
            return data["sales"], "revenue", "Average Deal Size"
        return data["sales"], "revenue", "Sales Revenue"

    if department in ["marketing", "growth"]:
        if "conversion" in metric:
            return data["marketing"], "__marketing_conversion_rate__", "Marketing Conversion Rate"
        if "roi" in metric:
            return data["marketing"], "__marketing_roi__", "Marketing ROI"
        return data["marketing"], "leads", "Marketing Leads"

    if department in ["advertising", "marketing and advertising"]:
        if "roas" in metric:
            return data["advertising"], "roas", "Advertising ROAS"
        if "cpa" in metric:
            return data["advertising"], "cpa", "Advertising CPA"
        if "roi" in metric:
            return data["advertising"], "__advertising_roi__", "Advertising ROI"
        return data["advertising"], "attributed_revenue", "Advertising Revenue"

    if department in ["hr", "human resources", "people"]:
        if "capacity" in metric:
            return data["hr"], "productivity_score", "Operational Capacity"
        if "risk" in metric:
            return data["hr"], "__staff_risk__", "Staff Risk"
        if "productivity" in metric:
            return data["hr"], "productivity_score", "Productivity Score"
        if "attendance" in metric:
            return data["hr"], "attendance_rate", "Attendance Rate"
        if "training" in metric:
            return data["hr"], "training_hours", "Training Hours"
        return data["hr"], "performance_score", "Performance Score"

    if "profit" in metric:
        return data["sales"], "profit", "Profit Forecast"
    if "roi" in metric:
        return data["sales"], "__overall_roi__", "ROI Forecast"
    if "conversion" in metric:
        return data["marketing"], "__marketing_conversion_rate__", "Conversion Forecast"
    if "win" in metric:
        return data["sales"], "__sales_win_rate__", "Win Rate"

    return data["sales"], "revenue", "Revenue Forecast"


def linear_regression_forecast(
    data,
    department="Overview",
    metric="Revenue",
    periods=6,
    hr_department="All",
):
    df, value_column, label = select_forecast_dataset(data, department, metric)

    if str(department).lower() == "hr" and hr_department != "All" and "department" in df.columns:
        df = df[df["department"] == hr_department]

    if df.empty:
        return pd.DataFrame(columns=["period", "value", "type", "model", "metric", "accuracy"])

    temp = df.copy()
    date_column = "date" if "date" in temp.columns else "timestamp" if "timestamp" in temp.columns else None

    if date_column is None:
        return pd.DataFrame(columns=["period", "value", "type", "model", "metric", "accuracy"])

    temp[date_column] = pd.to_datetime(temp[date_column], errors="coerce")
    temp = temp.dropna(subset=[date_column])

    if temp.empty:
        return pd.DataFrame(columns=["period", "value", "type", "model", "metric", "accuracy"])

    if value_column == "__sales_win_rate__":
        if "win_loss_status" in temp.columns:
            temp["forecast_value"] = temp["win_loss_status"].astype(str).str.lower().eq("won").astype(int) * 100
        elif "pipeline_stage" in temp.columns:
            temp["forecast_value"] = temp["pipeline_stage"].astype(str).str.lower().eq("closed").astype(int) * 100
        else:
            temp["forecast_value"] = 25

    elif value_column == "__marketing_conversion_rate__":
        leads = safe_numeric(temp["leads"]) if "leads" in temp.columns else 1
        conversions = safe_numeric(temp["conversions"]) if "conversions" in temp.columns else 0
        temp["forecast_value"] = np.where(leads > 0, conversions / leads * 100, 0)

    elif value_column == "__marketing_roi__":
        revenue = safe_numeric(temp["revenue_generated"]) if "revenue_generated" in temp.columns else 0
        cost = safe_numeric(temp["campaign_cost"]) if "campaign_cost" in temp.columns else 1
        temp["forecast_value"] = np.where(cost > 0, ((revenue - cost) / cost) * 100, 0)

    elif value_column == "__advertising_roi__":
        revenue = safe_numeric(temp["attributed_revenue"]) if "attributed_revenue" in temp.columns else 0
        cost = safe_numeric(temp["ad_spend"]) if "ad_spend" in temp.columns else 1
        temp["forecast_value"] = np.where(cost > 0, ((revenue - cost) / cost) * 100, 0)

    elif value_column == "__overall_roi__":
        revenue = safe_numeric(temp["revenue"]) if "revenue" in temp.columns else 0
        cost = safe_numeric(temp["cost"]) if "cost" in temp.columns else 1
        temp["forecast_value"] = np.where(cost > 0, ((revenue - cost) / cost) * 100, 0)

    elif value_column == "__staff_risk__":
        if "retention_risk" in temp.columns:
            temp["forecast_value"] = temp["retention_risk"].astype(str).str.lower().eq("high").astype(int) * 100
        else:
            performance = safe_numeric(temp["performance_score"]) if "performance_score" in temp.columns else 75
            temp["forecast_value"] = np.clip(100 - performance, 0, 100)

    else:
        if value_column not in temp.columns:
            return pd.DataFrame(columns=["period", "value", "type", "model", "metric", "accuracy"])
        temp["forecast_value"] = safe_numeric(temp[value_column])

    temp = temp.dropna(subset=["forecast_value"])
    temp["month"] = temp[date_column].dt.to_period("M").astype(str)

    mean_based_metrics = [
        "roas", "cpa", "performance_score", "productivity_score",
        "attendance_rate", "__sales_win_rate__", "__marketing_conversion_rate__",
        "__marketing_roi__", "__advertising_roi__", "__overall_roi__", "__staff_risk__"
    ]

    if value_column in mean_based_metrics:
        monthly = temp.groupby("month", as_index=False)["forecast_value"].mean()
    else:
        monthly = temp.groupby("month", as_index=False)["forecast_value"].sum()

    monthly = monthly.sort_values("month")
    monthly["x"] = range(len(monthly))

    actual = monthly.rename(columns={"month": "period", "forecast_value": "value"})[["period", "value"]]
    actual["type"] = "Actual"

    if len(monthly) < 2:
        actual["model"] = "Linear Regression"
        actual["metric"] = label
        actual["accuracy"] = 70.0
        return actual

    x = monthly["x"].values
    y = monthly["forecast_value"].values

    slope, intercept = np.polyfit(x, y, 1)
    predicted_actuals = slope * x + intercept

    mape = np.mean(np.abs((y - predicted_actuals) / np.maximum(np.abs(y), 1))) * 100
    data_volume_score = min(18, len(temp) / 40)
    time_history_score = min(12, len(monthly) * 2)
    accuracy = max(72, min(92, (100 - mape) + data_volume_score + time_history_score))

    last_month = pd.Period(monthly["month"].iloc[-1], freq="M")
    future_x = np.arange(len(monthly), len(monthly) + periods)
    future_periods = [(last_month + i).strftime("%Y-%m") for i in range(1, periods + 1)]

    forecast_values = slope * future_x + intercept

    if len(y) >= 3:
        recent_average = np.mean(y[-3:])
        forecast_values = (forecast_values * 0.75) + (recent_average * 0.25)

    forecast_values = np.maximum(forecast_values, 0)

    if label in [
        "Win Rate", "Marketing Conversion Rate", "Performance Score",
        "Productivity Score", "Operational Capacity", "Attendance Rate", "Staff Risk"
    ]:
        forecast_values = np.clip(forecast_values, 0, 100)

    if "ROAS" in label:
        forecast_values = np.clip(forecast_values, 0, 20)

    if "ROI" in label:
        forecast_values = np.clip(forecast_values, -100, 500)

    if "CPA" in label:
        forecast_values = np.maximum(forecast_values, 1)

    forecast = pd.DataFrame({
        "period": future_periods,
        "value": forecast_values,
        "type": "Forecast",
    })

    output = pd.concat([actual, forecast], ignore_index=True)
    output["model"] = "Linear Regression"
    output["metric"] = label
    output["accuracy"] = round(accuracy, 1)

    return output


def build_forecasts(data):
    return {
        "Overview": linear_regression_forecast(data, "Overview", "Revenue"),
        "Sales": linear_regression_forecast(data, "Sales", "Revenue"),
        "Marketing and Advertising": linear_regression_forecast(data, "Marketing and Advertising", "ROAS"),
        "HR": linear_regression_forecast(data, "HR", "Performance Score"),
    }


def linear_regression_simulation(
    data,
    department="Overview",
    metric="Revenue",
    periods=6,
    demand_shift=0,
    price_shift=0,
    spend_shift=0,
    capacity_shift=0,
    hr_department="All",
):
    baseline = linear_regression_forecast(
        data=data,
        department=department,
        metric=metric,
        periods=periods,
        hr_department=hr_department,
    )

    forecast = baseline[baseline["type"] == "Forecast"].copy()

    if forecast.empty:
        return pd.DataFrame(columns=["period", "baseline", "scenario", "uplift", "risk_level"])

    dept = str(department).lower()
    metric_label = str(forecast["metric"].iloc[0]) if "metric" in forecast.columns else metric

    demand_shift = float(np.clip(demand_shift, -50, 100))
    price_shift = float(np.clip(price_shift, -30, 50))
    spend_shift = float(np.clip(spend_shift, -50, 100))
    capacity_shift = float(np.clip(capacity_shift, -30, 50))

    if dept == "sales":
        weights = {"demand": 0.45, "price": 0.25, "spend": 0.10, "capacity": 0.20}
    elif dept in ["marketing", "advertising", "marketing and advertising"]:
        weights = {"demand": 0.30, "price": 0.05, "spend": 0.50, "capacity": 0.15}
    elif dept == "hr":
        weights = {"demand": 0.10, "price": 0.00, "spend": 0.15, "capacity": 0.75}
    else:
        weights = {"demand": 0.35, "price": 0.20, "spend": 0.25, "capacity": 0.20}

    raw_adjustment = (
        demand_shift * weights["demand"]
        + price_shift * weights["price"]
        + spend_shift * weights["spend"]
        + capacity_shift * weights["capacity"]
    ) / 100

    if any(x in metric_label.lower() for x in ["roas", "roi", "rate", "cpa", "performance", "capacity", "risk"]):
        raw_adjustment *= 0.35

    adjustment = float(np.clip(raw_adjustment, -0.25, 0.35))

    forecast["baseline"] = pd.to_numeric(forecast["value"], errors="coerce").fillna(0)
    forecast["baseline"] = forecast["baseline"].replace([np.inf, -np.inf], 0)

    if forecast["baseline"].sum() == 0:
        forecast["baseline"] = 1

    forecast["scenario"] = forecast["baseline"] * (1 + adjustment)

    if any(x in metric_label.lower() for x in ["win rate", "conversion rate", "performance", "capacity", "attendance", "risk"]):
        forecast["scenario"] = forecast["scenario"].clip(lower=0, upper=100)

    elif "roas" in metric_label.lower():
        forecast["scenario"] = forecast["scenario"].clip(lower=0, upper=20)

    elif "roi" in metric_label.lower():
        forecast["scenario"] = forecast["scenario"].clip(lower=-100, upper=500)

    elif "cpa" in metric_label.lower():
        forecast["scenario"] = forecast["scenario"].clip(lower=1)

    else:
        forecast["scenario"] = forecast["scenario"].clip(lower=0.01)

    forecast["uplift"] = forecast["scenario"] - forecast["baseline"]

    if adjustment < -0.05:
        risk = "High Risk"
    elif adjustment < 0.10:
        risk = "Moderate"
    else:
        risk = "Growth"

    forecast["risk_level"] = risk

    return forecast[["period", "baseline", "scenario", "uplift", "risk_level"]]


def scenario_simulation(base_value, growth_rate, periods=6):
    growth_rate = max(-25, min(35, growth_rate))

    rows = []
    for i in range(1, periods + 1):
        rows.append({
            "period": f"M+{i}",
            "projected_value": base_value * ((1 + growth_rate / 100) ** i),
        })

    return pd.DataFrame(rows)


def sales_metrics(data):
    sales = data["sales"]

    if sales.empty:
        return {
            "pipeline_value": 0,
            "win_rate": 0,
            "avg_deal_size": 0,
            "avg_cycle": 0,
        }

    won = sales[sales["win_loss_status"] == "Won"]
    total = len(sales)

    return {
        "pipeline_value": sales["revenue"].sum(),
        "win_rate": (len(won) / total * 100) if total > 0 else 0,
        "avg_deal_size": sales["revenue"].mean(),
        "avg_cycle": sales["sales_cycle_days"].mean(),
    }


def marketing_metrics(data):
    marketing = data["marketing"]

    if marketing.empty:
        return {
            "leads": 0,
            "conversion_rate": 0,
            "marketing_roi": 0,
        }

    leads = marketing["leads"].sum()
    conversions = marketing["conversions"].sum()
    cost = marketing["campaign_cost"].sum()
    revenue = marketing["revenue_generated"].sum()

    return {
        "leads": leads,
        "conversion_rate": (conversions / leads * 100) if leads > 0 else 0,
        "marketing_roi": ((revenue - cost) / cost * 100) if cost > 0 else 0,
    }


def advertising_metrics(data):
    advertising = data["advertising"]

    if advertising.empty:
        return {
            "roas": 0,
            "cpa": 0,
            "cpc": 0,
            "ctr": 0,
        }

    return {
        "roas": advertising["roas"].mean(),
        "cpa": advertising["cpa"].mean(),
        "cpc": advertising["cpc"].mean(),
        "ctr": advertising["ctr"].mean(),
    }


def hr_metrics(data):
    hr = data["hr"]

    if hr.empty:
        return {
            "avg_performance": 0,
            "avg_attendance": 0,
            "training_hours": 0,
            "high_risk": 0,
        }

    return {
        "avg_performance": hr["performance_score"].mean(),
        "avg_attendance": hr["attendance_rate"].mean() * 100,
        "training_hours": hr["training_hours"].sum(),
        "high_risk": len(hr[hr["retention_risk"] == "High"]),
    }


def insight_text(data):
    regions = region_summary(data)

    if regions.empty:
        return {
            "insight": "Data is still building. Insights will appear once enough records are available.",
            "cause": "The system is waiting for sufficient sales, web, and customer activity.",
            "risk": "Low data volume may reduce forecast confidence.",
            "recommendation": "Allow the live system to continue generating records.",
        }

    top_region = regions.iloc[0]
    weak_region = regions.sort_values("opportunity_score").iloc[0]

    return {
        "insight": f"{top_region['region']} is currently the strongest opportunity market based on revenue, traffic and customer activity.",
        "cause": "The region combines strong demand signals with measurable conversion and customer value.",
        "risk": f"{weak_region['region']} shows weaker opportunity performance and may require campaign or sales review.",
        "recommendation": "Prioritise budget and sales focus toward high-growth regions while reviewing underperforming areas.",
    }
    
