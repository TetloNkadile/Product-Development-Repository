from pathlib import Path
from datetime import datetime, timedelta
import random
import csv
import json
import uuid

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

SALES_FILE = DATA_DIR / "sales_data.csv"
MARKETING_FILE = DATA_DIR / "marketing_data.csv"
ADVERTISING_FILE = DATA_DIR / "advertising_data.csv"
HR_FILE = DATA_DIR / "hr_data.csv"
CUSTOMER_FILE = DATA_DIR / "customer_data.csv"
WEB_LOG_FILE = DATA_DIR / "web_logs.json"
SYSTEM_LOG_FILE = DATA_DIR / "system_activity.log"

LOCATIONS = [
    {"country": "Botswana", "region": "Gaborone", "lat": -24.6282, "lon": 25.9231, "market_status": "Existing"},
    {"country": "Botswana", "region": "Francistown", "lat": -21.1702, "lon": 27.5079, "market_status": "Growth"},
    {"country": "Botswana", "region": "Maun", "lat": -19.9833, "lon": 23.4167, "market_status": "Expansion"},
    {"country": "South Africa", "region": "Johannesburg", "lat": -26.2041, "lon": 28.0473, "market_status": "Existing"},
    {"country": "South Africa", "region": "Cape Town", "lat": -33.9249, "lon": 18.4241, "market_status": "Growth"},
    {"country": "Namibia", "region": "Windhoek", "lat": -22.5609, "lon": 17.0658, "market_status": "Expansion"},
    {"country": "Zambia", "region": "Lusaka", "lat": -15.3875, "lon": 28.3228, "market_status": "Potential"},
    {"country": "Zimbabwe", "region": "Harare", "lat": -17.8292, "lon": 31.0522, "market_status": "Potential"},
    {"country": "Kenya", "region": "Nairobi", "lat": -1.2921, "lon": 36.8219, "market_status": "Potential"},
    {"country": "Nigeria", "region": "Lagos", "lat": 6.5244, "lon": 3.3792, "market_status": "Potential"},
]

SERVICES = [
    "BI Dashboard",
    "AI Chatbot",
    "Data Automation",
    "Cybersecurity",
    "Forecasting Model",
    "HR Analytics",
    "Process Optimisation",
]

CHANNELS = ["Organic", "Google Ads", "LinkedIn", "Facebook", "Referral", "Email"]
PLATFORMS = ["Google Ads", "LinkedIn Ads", "Facebook Ads", "Instagram Ads", "YouTube Ads"]
SEGMENTS = ["Enterprise", "SME", "Government", "Startup", "NGO"]
INDUSTRIES = ["Mining", "Finance", "Insurance", "Agriculture", "Retail", "Education", "Technology"]
CAMPAIGNS = ["Growth Intelligence", "AI Awareness", "Digital Efficiency", "Lead Generation", "Market Expansion"]
SALESPEOPLE = ["Neo", "Thato", "Aisha", "Kabelo", "Lerato", "Boitumelo", "Mpho"]
DEPARTMENTS = ["Sales", "Marketing", "Advertising", "HR", "Operations", "Analytics"]
ROLES = ["Analyst", "Officer", "Manager", "Specialist", "Coordinator", "Executive"]


def append_csv(path, fieldnames, row):
    file_exists = path.exists()

    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def load_json(path):
    if not path.exists():
        return []

    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError:
        return []


def save_json(path, records):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(records, file, indent=2)


def write_system_log(message):
    with open(SYSTEM_LOG_FILE, "a", encoding="utf-8") as file:
        file.write(f"{datetime.now().isoformat(timespec='seconds')} | {message}\n")


def get_date(history=False):
    if history:
        days_back = random.randint(1, 180)
        return (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    return datetime.now().strftime("%Y-%m-%d")


def seed_hr_data():
    if HR_FILE.exists():
        return

    hr_fields = [
        "employee_id",
        "date",
        "name",
        "department",
        "region",
        "role",
        "training_hours",
        "attendance_rate",
        "workload_score",
        "productivity_score",
        "performance_score",
        "revenue_contribution",
        "retention_risk",
    ]

    names = [
        "Neo",
        "Thato",
        "Aisha",
        "Kabelo",
        "Lerato",
        "Boitumelo",
        "Mpho",
        "Naledi",
        "Kagiso",
        "Tebogo",
        "Amantle",
        "Tshepo",
    ]

    for index, name in enumerate(names, start=1):
        department = random.choice(DEPARTMENTS)

        row = {
            "employee_id": f"E-{index:03d}",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "name": name,
            "department": department,
            "region": random.choice([location["region"] for location in LOCATIONS]),
            "role": random.choice(ROLES),
            "training_hours": random.randint(5, 100),
            "attendance_rate": round(random.uniform(0.75, 0.99), 2),
            "workload_score": round(random.uniform(2.0, 5.0), 2),
            "productivity_score": round(random.uniform(2.0, 5.0), 2),
            "performance_score": round(random.uniform(2.2, 5.0), 2),
            "revenue_contribution": round(random.uniform(0, 220000), 2) if department == "Sales" else 0,
            "retention_risk": random.choice(["Low", "Medium", "High"]),
        }

        append_csv(HR_FILE, hr_fields, row)


def generate_tick(rows_per_tick=8, history=False):
    seed_hr_data()

    sales_fields = [
        "sale_id",
        "date",
        "client_id",
        "campaign_id",
        "country",
        "region",
        "latitude",
        "longitude",
        "market_status",
        "service",
        "salesperson",
        "segment",
        "revenue",
        "cost",
        "profit",
        "pipeline_stage",
        "win_loss_status",
        "sales_cycle_days",
    ]

    marketing_fields = [
        "campaign_id",
        "date",
        "country",
        "region",
        "latitude",
        "longitude",
        "service",
        "campaign",
        "channel",
        "segment",
        "impressions",
        "clicks",
        "leads",
        "conversions",
        "campaign_cost",
        "revenue_generated",
    ]

    advertising_fields = [
        "ad_id",
        "campaign_id",
        "date",
        "country",
        "region",
        "latitude",
        "longitude",
        "service",
        "platform",
        "campaign",
        "impressions",
        "clicks",
        "conversions",
        "ad_spend",
        "attributed_revenue",
    ]

    customer_fields = [
        "client_id",
        "date",
        "country",
        "region",
        "latitude",
        "longitude",
        "segment",
        "industry",
        "service_interest",
        "lead_source",
        "conversion_status",
        "repeat_customer",
        "lifetime_value",
    ]

    web_logs = load_json(WEB_LOG_FILE)

    for _ in range(rows_per_tick):
        location = random.choice(LOCATIONS)
        service = random.choice(SERVICES)
        channel = random.choice(CHANNELS)
        segment = random.choice(SEGMENTS)
        campaign = random.choice(CAMPAIGNS)

        campaign_id = f"CAM-{random.randint(100, 999)}"
        client_id = f"C-{random.randint(1000, 9999)}"
        date_value = get_date(history=history)

        impressions = random.randint(1500, 85000)
        clicks = max(1, int(impressions * random.uniform(0.015, 0.12)))
        leads = max(1, int(clicks * random.uniform(0.08, 0.36)))
        conversions = max(1, int(leads * random.uniform(0.10, 0.62)))

        campaign_cost = round(random.uniform(1500, 40000), 2)
        revenue_generated = round(conversions * random.uniform(1800, 13000), 2)

        append_csv(MARKETING_FILE, marketing_fields, {
            "campaign_id": campaign_id,
            "date": date_value,
            "country": location["country"],
            "region": location["region"],
            "latitude": location["lat"],
            "longitude": location["lon"],
            "service": service,
            "campaign": campaign,
            "channel": channel,
            "segment": segment,
            "impressions": impressions,
            "clicks": clicks,
            "leads": leads,
            "conversions": conversions,
            "campaign_cost": campaign_cost,
            "revenue_generated": revenue_generated,
        })

        if channel in ["Google Ads", "LinkedIn", "Facebook"] or random.random() < 0.60:
            ad_spend = round(random.uniform(2500, 52000), 2)
            ad_conversions = max(1, int(clicks * random.uniform(0.03, 0.22)))
            attributed_revenue = round(ad_conversions * random.uniform(2500, 16000), 2)

            append_csv(ADVERTISING_FILE, advertising_fields, {
                "ad_id": f"AD-{random.randint(1000, 9999)}",
                "campaign_id": campaign_id,
                "date": date_value,
                "country": location["country"],
                "region": location["region"],
                "latitude": location["lat"],
                "longitude": location["lon"],
                "service": service,
                "platform": random.choice(PLATFORMS),
                "campaign": campaign,
                "impressions": impressions,
                "clicks": clicks,
                "conversions": ad_conversions,
                "ad_spend": ad_spend,
                "attributed_revenue": attributed_revenue,
            })

        conversion_status = random.choices(
            ["Converted", "Lead", "Not Converted"],
            weights=[38, 44, 18],
        )[0]

        lifetime_value = (
            round(random.uniform(10000, 260000), 2)
            if conversion_status == "Converted"
            else round(random.uniform(800, 18000), 2)
        )

        append_csv(CUSTOMER_FILE, customer_fields, {
            "client_id": client_id,
            "date": date_value,
            "country": location["country"],
            "region": location["region"],
            "latitude": location["lat"],
            "longitude": location["lon"],
            "segment": segment,
            "industry": random.choice(INDUSTRIES),
            "service_interest": service,
            "lead_source": channel,
            "conversion_status": conversion_status,
            "repeat_customer": random.choice(["Yes", "No"]),
            "lifetime_value": lifetime_value,
        })

        if conversion_status == "Converted" and random.random() < 0.72:
            revenue = round(random.uniform(8000, 130000), 2)
            cost = round(revenue * random.uniform(0.30, 0.72), 2)

            append_csv(SALES_FILE, sales_fields, {
                "sale_id": f"S-{random.randint(10000, 99999)}",
                "date": date_value,
                "client_id": client_id,
                "campaign_id": campaign_id,
                "country": location["country"],
                "region": location["region"],
                "latitude": location["lat"],
                "longitude": location["lon"],
                "market_status": location["market_status"],
                "service": service,
                "salesperson": random.choice(SALESPEOPLE),
                "segment": segment,
                "revenue": revenue,
                "cost": cost,
                "profit": round(revenue - cost, 2),
                "pipeline_stage": random.choice(["Lead", "Qualified", "Proposal", "Negotiation", "Closed"]),
                "win_loss_status": random.choice(["Won", "Pending", "Lost"]),
                "sales_cycle_days": random.randint(3, 75),
            })

        web_logs.append({
            "log_id": f"LOG-{uuid.uuid4().hex[:8].upper()}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "client_id": client_id,
            "campaign_id": campaign_id,
            "country": location["country"],
            "region": location["region"],
            "latitude": location["lat"],
            "longitude": location["lon"],
            "service_viewed": service,
            "traffic_source": channel,
            "device_type": random.choice(["Mobile", "Desktop", "Tablet"]),
            "page_views": random.randint(1, 10),
            "bounce_rate": round(random.uniform(0.12, 0.78), 2),
            "status_code": random.choices([200, 404, 500], weights=[91, 6, 3])[0],
            "response_time": round(random.uniform(0.10, 2.8), 2),
            "session_id": f"SID-{uuid.uuid4().hex[:8].upper()}",
        })

    save_json(WEB_LOG_FILE, web_logs[-5000:])
    write_system_log(f"Generated {rows_per_tick} {'historic' if history else 'current'} records.")


def initialise_data():
    seed_hr_data()

    files_exist = (
        SALES_FILE.exists()
        and MARKETING_FILE.exists()
        and ADVERTISING_FILE.exists()
        and CUSTOMER_FILE.exists()
        and WEB_LOG_FILE.exists()
    )

    if files_exist:
        return

    for _ in range(120):
        generate_tick(rows_per_tick=8, history=True)

    write_system_log("Historic dataset initialised.")


if __name__ == "__main__":
    initialise_data()
    generate_tick(rows_per_tick=8, history=False)
    print("Data generated successfully.")