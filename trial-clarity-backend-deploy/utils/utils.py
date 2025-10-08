import json, os, base64, uuid, asyncio, traceback
import pymc as pm
import numpy as np
import arviz as az
import pandas as pd
import seaborn as sns
import pingouin as pg
from io import BytesIO
import pymc_bart as pmb
import scipy.stats as stats
from typing import Optional
import redis.asyncio as redis
from sqlalchemy import insert
import matplotlib.pyplot as plt
from types import SimpleNamespace
from itertools import combinations
import xml.etree.ElementTree as ET
from collections import defaultdict
from lifelines import KaplanMeierFitter
from fastapi import Depends, HTTPException
from sqlmodel import Session, select
from lifelines.statistics import logrank_test
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.power import TTestIndPower
from datetime import datetime, timezone, timedelta
from fastapi import WebSocket, WebSocketDisconnect
from scipy.stats import chi2_contingency, norm, spearmanr

from ..core.database import get_db_session
from ..utils.queue_manager import queue_manager
from   ..apis.v1.schemas import TrialEndPointResponse
from ..models.core import TrialFile, TrialData, Trial, TrialResult, TrialEndPoint


async def bayes_factor_by_unc(p_unc):
    # Step 1: Compute the z-score from the p-value (two-tailed test)
    z_score = stats.norm.ppf(1 - p_unc / 2)

    # Step 2: Compute BF01 using Wagenmakers' approximation
    BF01 = np.exp(-0.5 * z_score**2) / (np.sqrt(2 * np.pi) * z_score)

    # Step 3: Compute BF10 (which is 1/BF01)
    BF10 = 1 / BF01

    return {
        "z-score": z_score,
        "BF01": BF01,
        "BF10": BF10
    }

def clean_data_for_json(data):
    """Recursively clean data for JSON serialization"""
    if isinstance(data, (str, int, bool, type(None))):
        return data
    elif isinstance(data, float):
        return None if np.isinf(data) or pd.isna(data) else data  # Handle NaN and inf
    elif isinstance(data, dict):
        return {k: clean_data_for_json(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [clean_data_for_json(item) for item in data]
    else:
        return data  # Fallback for other types


async def generate_weekly_cumulative_ranges(filtered_dataframe: pd.DataFrame, is_weekly_cumilative: Optional[bool] = None):
    start_date = pd.to_datetime(filtered_dataframe["date"].min()).date()
    end_date = pd.to_datetime(filtered_dataframe["date"].max()).date()

    if is_weekly_cumilative == False:
        return [
            {
            "start_date": start_date,
            "end_date": end_date,
            "week_start_date": start_date
            }
        ]

    weekly_ranges = []

    def get_week_start_date(end_date: datetime.date):
        week_start = end_date - timedelta(days=end_date.weekday())  # Monday of that week
        return max(start_date, week_start)

    days_until_sunday = (6 - start_date.weekday()) % 7
    current_end = start_date + timedelta(days=days_until_sunday)

    seen_counts = set()

    while current_end < end_date:
        current_data = filtered_dataframe[
            (filtered_dataframe["date"] >= start_date) &
            (filtered_dataframe["date"] <= current_end)
        ]
        count = len(current_data)

        if count and count not in seen_counts:
            week_start = get_week_start_date(current_end)
            weekly_ranges.append({
                "start_date": start_date,
                "end_date": current_end,
                "week_start_date": week_start
            })
            seen_counts.add(count)

        current_end += timedelta(days=7)

    final_data = filtered_dataframe[
        (filtered_dataframe["date"] >= start_date) &
        (filtered_dataframe["date"] <= end_date)
    ]
    final_count = len(final_data)
    if final_count and final_count not in seen_counts:
        week_start = get_week_start_date(end_date)
        weekly_ranges.append({
            "start_date": start_date,
            "end_date": end_date,
            "week_start_date": week_start
        })
    return weekly_ranges

def convert_numpy_types(obj):
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    return obj

def sanitize_bf_value(val):
    if np.isinf(val):
        return 1e308
    elif np.isnan(val):
        return None
    return val
    

def encode_image(fig_buffer):
    fig_buffer.seek(0)
    return base64.b64encode(fig_buffer.getvalue()).decode('utf-8')

# async def create_trial_data(trial_file_id, df, db: Session = Depends(get_db_session)):
#     records = []
#     for index, row in df.iterrows():
#         for column, value in row.items():
#             record = TrialData(trial_file_id=trial_file_id, row=index, attribute_key=column, attribute_value=str(value))
#             records.append(record)
#     db.add_all(records)
#     db.commit()
#     return True


async def create_trial_data(trial_file_id: int, df: pd.DataFrame, db: Session = Depends(get_db_session)): #Completed in 6.89 seconds
    now = datetime.now(timezone.utc)

    # Flatten the DataFrame (row, attribute_key, attribute_value)
    df_melted = df.reset_index().melt(
        id_vars='index',
        var_name='attribute_key',
        value_name='attribute_value'
    )
    df_melted.rename(columns={'index': 'row'}, inplace=True)

    # Required fields from BaseModel and RawBaseModel
    base_fields = {
        'status': True,
        'deleted': False,
        'created_at': now,
        'updated_at': now,
        'deleted_at': None,
        'created_by': None,
        'updated_by': None,
        'deleted_by': None,
    }

    # Convert to list of dicts for bulk insert
    records = []
    for _, row in df_melted.iterrows():
        record = {
            'trial_file_id': trial_file_id,
            'row': int(row['row']),
            'attribute_key': str(row['attribute_key']),
            'attribute_value': str(row['attribute_value']),
            **base_fields
        }
        records.append(record)

    # Perform bulk insert using SQLAlchemy Core
    stmt = insert(TrialData)
    db.execute(stmt, records)
    db.commit()

    return True

def convert_value(value: str):
    if value.isdigit():
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value

async def parse_xml(xml_content: str | bytes):
    try:
        root = xml_content if isinstance(xml_content, ET.Element) else ET.fromstring(xml_content)

        # Get headers in order of first occurrence
        headers = []
        seen_headers = set()
        for row in root.findall(".//row_data"):
            for cell in row:
                tag = cell.tag.strip()
                if tag not in seen_headers:
                    seen_headers.add(tag)
                    headers.append(tag)
        
        rows = [
            {
                cell.tag.strip(): convert_value(cell.text.strip()) if cell.text else ""
                for cell in row
            }
            for row in root.findall(".//row_data")
        ]
        
        return pd.DataFrame(rows, columns=headers) if rows else pd.DataFrame()
    except Exception as e:
        raise ValueError(f"Error parsing XML: {e}")

async def parse_json(contents):
    try:
        data = json.loads(contents)
        return pd.DataFrame(data)
    except (json.JSONDecodeError, KeyError) as e:
        raise ValueError(f"Error parsing JSON: {e}")

def convert(obj):
    if isinstance(obj, np.ndarray):  # Convert NumPy arrays to lists
        return obj.tolist()
    elif isinstance(obj, np.float32) or isinstance(obj, np.float64):  # Convert NumPy floats
        return float(obj)
    elif isinstance(obj, np.int32) or isinstance(obj, np.int64):  # Convert NumPy integers
        return int(obj)
    return obj

def replace_nan_with_none(d):
            if isinstance(d, dict):
                return {k: replace_nan_with_none(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [replace_nan_with_none(v) for v in d]
            elif isinstance(d, (np.float64, float)) and np.isnan(d):
                return None
            return d

async def create_trial_file(organisation_id: int, file_name: str, df: pd.DataFrame, file_type: str | None, db: Session = Depends(get_db_session)):
    for col in df.select_dtypes(include=["datetime64[ns]"]).columns:
        df[col] = df[col].apply(lambda x: x.isoformat() if pd.notnull(x) else None)

    unique_headers = []
    for idx, col in enumerate(df.columns):
        col_data = df[col].infer_objects()
        non_null_types = col_data.dropna().map(type).unique()
        if str in non_null_types:
            values_type = "str"
        elif len(non_null_types) > 0:
            values_type = non_null_types[0].__name__
        else:
            values_type = "NoneType"
        
        if values_type == "NoneType":
            type_counts = df[col].astype(str).map(type).value_counts().index.tolist()
            values_type = next((t.__name__ for t in type_counts if t.__name__ != "NoneType"), "NoneType")

        unique_headers.append({
            "name": col,
            "values_type": values_type,
            "order": idx
        })

    existing_trial = db.exec(
    select(TrialFile.trial_id, Trial.name)
    .join(Trial, Trial.id == TrialFile.trial_id)
        .where(Trial.organisation_id == organisation_id, TrialFile.file_headers == unique_headers, TrialFile.status == True, TrialFile.deleted == False)).first()

    if existing_trial:
        trial_id, trial_name = existing_trial
    else:
        trial_name = f"newfile_{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M')}"
        new_trial = Trial(
            organisation_id=organisation_id,
            name=trial_name,
            start_date=datetime.utcnow(),
            end_date=datetime.utcnow(),
            control_preset=False,
            randomized=False,
            blinded=False
        )
        db.add(new_trial)
        db.commit()
        db.refresh(new_trial)
        trial_id = new_trial.id
        trial_name = new_trial.name
   
    row_count = df.shape[0]

    trial_file = TrialFile(
        trial_id=trial_id,
        name=file_name,
        row_count=row_count,
        file_type=file_type,
        data=df.to_dict(orient="records"),
        file_headers=unique_headers
    )

    db.add(trial_file)
    db.commit()
    db.refresh(trial_file)

    return {
        "trial_file": trial_file,
        "unique_headers": unique_headers,
        "row_count": row_count,
        "trial_name": trial_name,
        "trial_id": trial_id
    }

async def create_trial_result(data: dict):
    trial_record = data["trial_record"]
    trial_req = data["trial_req"]
    bayesian_value = data["bayesian_value"]
    result = data["result"]
    system_generation_data = data.get("system_generation_data", {})
    selected_columns = data["selected_columns"]
    avg_validation = data.get("avg_validation")
    avg_result = data.get("avg_result")
    rec_to_stop = data.get("rec_to_stop")
    db: Session = data["db"]

    is_system_generation = bool(system_generation_data)
    start_date = system_generation_data.get('start_date') if is_system_generation else trial_record.start_date
    end_date = system_generation_data.get('end_date') if is_system_generation else trial_record.end_date
    week_start_date = system_generation_data.get('week_start_date') if is_system_generation else None
    endpoint_id = trial_req.endpoint_id if is_system_generation else None

    trial_result = TrialResult(
        trial_id=trial_record.id,
        start_date=start_date,
        end_date=end_date,
        week_start_date=week_start_date,
        columns=selected_columns,
        bayesian_value=bayesian_value,
        is_system_generated=is_system_generation,
        endpoint_id=endpoint_id,
        result=result,
        avg_validation=avg_validation,
        rec_to_stop = rec_to_stop,
        avg_result=avg_result
    )

    db.add(trial_result)
    db.commit()
    db.refresh(trial_result)
    return trial_result.id

async def get_bayes_message():
    data = [
            {
                "min": 0,
                "max": 0.01,
                "message": "extreme evidence that the two treatment groups are statistically different"
            },
            {
                "min": 0.01,
                "max": 0.03,
                "message": "very strong evidence that the two treatment groups are statistically different"
            },
            {
                "min": 0.03,
                "max": 0.1,
                "message": "strong evidence that the two treatment groups are statistically different"
            },
            {
                "min": 0.1,
                "max": 0.3,
                "message": "moderate evidence that the two treatment groups are statistically different"
            },
            {
                "min": 0.3,
                "max": 1,
                "message": "anecdotal evidence that the two treatment groups are statistically different"
            },
            {
                "min": 1,
                "max": 3,
                "message": "anecdotal evidence that the two treatment groups are statistically similar"
            },
            {
                "min": 3,
                "max": 10,
                "message": "moderate evidence that the two treatment groups are statistically similar"
            },
            {
                "min": 10,
                "max": 30,
                "message": "strong evidence that the two treatment groups are statistically similar"
            },
            {
                "min": 30,
                "max": 100,
                "message": "very strong evidence that the two treatment groups are statistically similar"
            },
            {
                "min": 100,
                "max": 10000000,
                "message": "extreme evidence that the two treatment groups are statistically similar"
            }
        ]
    return data

async def get_trial_rec_without_date_filter(file_records):
        processed_dataframes = []
        for file_record in file_records:
            if not file_record.file_headers or not file_record.data:
                continue
            parsed_data = await parse_json(json.dumps(file_record.data or {}))
            ordered_columns = [header["name"] for header in sorted(file_record.file_headers, key=lambda x: x["order"])]
            selected_dataframe = parsed_data[ordered_columns] # Select only the ordered columns
            selected_dataframe = selected_dataframe.assign(date=lambda df: pd.to_datetime(df["date"], errors="coerce").dt.date)
            processed_dataframes.append(selected_dataframe)

        combined_dataframe = pd.concat(processed_dataframes, ignore_index=True)
        filtered_dataframe = combined_dataframe.reset_index(drop=True)
        return filtered_dataframe


async def get_trial_rec(file_records, trial_record):
        processed_dataframes = []
        for file_record in file_records:
            if not file_record.file_headers or not file_record.data:
                continue

            parsed_data = await parse_json(json.dumps(file_record.data or {}))
            ordered_columns = [header["name"] for header in sorted(file_record.file_headers, key=lambda x: x["order"])]
            selected_dataframe = parsed_data[ordered_columns] # Select only the ordered columns
            selected_dataframe = selected_dataframe.assign(date=lambda df: pd.to_datetime(df["date"], errors="coerce").dt.date)
            processed_dataframes.append(selected_dataframe)

        combined_dataframe = pd.concat(processed_dataframes, ignore_index=True)
        filtered_dataframe = combined_dataframe.query("@trial_record.start_date.date() <= date <= @trial_record.end_date.date()", engine="python").reset_index(drop=True)
        return filtered_dataframe


async def get_custome_trial_rec(file_records, start_date, end_date):
        processed_dataframes = []
        for file_record in file_records:
            if not file_record.file_headers or not file_record.data:
                continue

            parsed_data = await parse_json(json.dumps(file_record.data or {}))
            ordered_columns = [header["name"] for header in sorted(file_record.file_headers, key=lambda x: x["order"])]
            selected_dataframe = parsed_data[ordered_columns] # Select only the ordered columns
            selected_dataframe = selected_dataframe.assign(date=lambda df: pd.to_datetime(df["date"], errors="coerce").dt.date)
            processed_dataframes.append(selected_dataframe)

        combined_dataframe = pd.concat(processed_dataframes, ignore_index=True)
        start_date = pd.to_datetime(start_date).date()
        end_date = pd.to_datetime(end_date).date()
        
        filtered_dataframe = combined_dataframe.query("@start_date <= date <= @end_date", engine="python").reset_index(drop=True)
        return filtered_dataframe


async def generate_weekly_result(trial_id, endpoint, db: Session = Depends(get_db_session)):
    processed_dataframes = []
    trial_record = db.exec(select(Trial).where(Trial.id ==trial_id, Trial.status == True, Trial.deleted == False)).first()
    if not trial_record:
        raise HTTPException(status_code=404, detail="Trial not found")

    file_records = db.exec(select(TrialFile).where(TrialFile.trial_id == trial_id, 
                                                    TrialFile.status == True, TrialFile.deleted == False)).all()
    
    for file_record in file_records:
        if not file_record.file_headers or not file_record.data:
            continue

        parsed_data = await parse_json(json.dumps(file_record.data or {}))
        ordered_columns = [header["name"] for header in sorted(file_record.file_headers, key=lambda x: x["order"])]
        selected_dataframe = parsed_data[ordered_columns] # Select only the ordered columns
        selected_dataframe = selected_dataframe.assign(date=lambda df: pd.to_datetime(df["date"], errors="coerce").dt.date)
        processed_dataframes.append(selected_dataframe)

    combined_dataframe = pd.concat(processed_dataframes, ignore_index=True)
    filtered_dataframe = combined_dataframe.reset_index(drop=True)
    weekly_cumulative_ranges = await generate_weekly_cumulative_ranges(filtered_dataframe, endpoint.is_weekly_cumilative)

    data = {
        "weekly_cumulative_ranges": weekly_cumulative_ranges,
        "trial_plot_request": {
            "trial_id": trial_id,
            "endpoint_id": endpoint.id,
            "columns": endpoint.column,
            "endpoint_data": TrialEndPointResponse.model_validate(endpoint,  from_attributes=True)
        }
    }

    await queue_manager.add_message(trial_id, data)
    return True


async def validate_entity(db, model, entity_id, error_msg, status_code=400):
        if entity_id and not db.query(model).filter_by(id=entity_id, status=True, deleted=False).first():
            raise HTTPException(status_code=status_code, detail=error_msg)


def v1_get_bayes_interp(case_number: int, value: float) -> str:
    if case_number == 3:
        return (
            "The two parameters are associated with each other"
            if value < 0.05
            else "The two parameters are not associated with each other"
        )

    if value == 1.0:
        if case_number == 2:
            return "No evidence that the event is a random (50% chance)"
        else:
            return "No evidence that the two groups are statistically similar"

    if case_number == 2:
        bayes_event_ranges = [
            ((0, 0.01), "Extreme evidence that the event is not random (not a 50% chance)"),
            ((0.01, 0.033), "Very strong evidence that the event is not random (not a 50% chance)"),
            ((0.033, 0.1), "Strong evidence that the event is not random (not a 50% chance)"),
            ((0.1, 0.33), "Moderate evidence that the event is not random (not a 50% chance)"),
            ((0.33, 1), "Anecdotal evidence that the event is not random (not a 50% chance)"),
            ((1.00000001, 3), "Anecdotal evidence that the event is a random (50% chance)"),
            ((3, 10), "Moderate evidence that the event is a random (50% chance)"),
            ((10, 30), "Strong evidence that the event is a random (50% chance)"),
            ((30, 100), "Very strong evidence that the event is a random (50% chance)"),
            ((100, float('inf')), "Extreme evidence that the event is a random (50% chance)"),
        ]
        for (lower, upper), interpretation in bayes_event_ranges:
            if lower <= value < upper:
                return interpretation

    standard_bayes_ranges = [
        ((0, 0.01), "Extreme evidence that the two groups are statistically different"),
        ((0.01, 0.033), "Very strong evidence that the two groups are statistically different"),
        ((0.033, 0.1), "Strong evidence that the two groups are statistically different"),
        ((0.1, 0.33), "Moderate evidence that the two groups are statistically different"),
        ((0.33, 1), "Anecdotal evidence that the two groups are statistically different"),
        ((1.00000001, 3), "Anecdotal evidence that the two groups are statistically similar"),
        ((3, 10), "Moderate evidence that the two groups are statistically similar"),
        ((10, 30), "Strong evidence that the two groups are statistically similar"),
        ((30, 100), "Very strong evidence that the two groups are statistically similar"),
        ((100, float('inf')), "Extreme evidence that the two groups are statistically similar"),
    ]
    for (lower, upper), interpretation in standard_bayes_ranges:
        if lower <= value < upper:
            return interpretation

    return "No interpretation available for the given value"


def get_bayes_interp(val):
    ranges = [[(0, 0.01), 'extreme evidence that the two treatment groups are statistically different'],
                [(0.01, 0.03), 'very strong evidence that the two treatment groups are statistically different'],
                [(0.03, 0.1), 'strong evidence that the two treatment groups are statistically different'],
                [(0.1, 0.3), 'moderate evidence that the two treatment groups are statistically different'],
                [(0.3, 1), 'anecdotal evidence that the two treatment groups are statistically different'],
                [(1, 3), 'anecdotal evidence that the two treatment groups are statistically similar'],
                [(3, 10), 'moderate evidence that the two treatment groups are statistically similar'],
                [(10, 30), 'strong evidence that the two treatment groups are statistically similar'],
                [(30, 100), 'very strong evidence that the two treatment groups are statistically similar'],
                [(100, 10000000), 'extreme evidence that the two treatment groups are statistically similar']
            ]
    for r in ranges:
        if ((val >= r[0][0]) and (val < r[0][1])):
            return r[1]
    return ranges[-1][1]


def interpolate_color(val):
    ranges = [[(0, 0.01), '#006400'],
                [(0.01, 0.03), '#338000'],
                [(0.03, 0.1), '#669900'],
                [(0.1, 0.3), '#99B300'],
                [(0.3, 1), '#CCCC00'],
                [(1, 3), '#FFCC00'],
                [(3, 10), '#FF9900'],
                [(10, 30), '#FF6600'],
                [(30, 100), '#993300'],
                [(100, 10000000), '#800000']
            ]


async def generate_graph_image(df_long):

    output_dir = "graphs_output"
    os.makedirs(output_dir, exist_ok=True)
    fig1, fig2, fig3, fig4  = BytesIO(), BytesIO(), BytesIO(), BytesIO()

    plt.figure(figsize=(12,9))
    sns.pointplot(df_long, x='time', y='value', hue='class', errorbar=lambda x: (x.quantile(0.25), x.quantile(0.75)), capsize=0.1, dodge=True, palette='bright')
    plt.title('Line + IQR over time')
    plt.savefig(os.path.join(output_dir, "pointplot.png"))
    plt.close()

    plt.figure(figsize=(12,9))
    sns.barplot(df_long, x='time', y='value', hue='class', errorbar=lambda x: (x.quantile(0.25), x.quantile(0.75)), capsize=0.1, palette='bright')
    plt.title('Bar + IQR over time')
    plt.savefig(os.path.join(output_dir, "Bar + IQR over time.png"))
    plt.close()

    plt.figure(figsize=(12,9))
    sns.boxplot(df_long, x='time', y='value', hue='class', fill=True, gap=0.1, palette='bright')
    plt.title('Box CI95 over time')
    plt.savefig(os.path.join(output_dir, "Box CI95 over time.png"))
    plt.close()

    plt.figure(figsize=(12,9))
    sns.stripplot(df_long, x='time', y='value', hue='class', dodge=True, jitter=0.25, legend=False, palette='bright')
    sns.boxplot(df_long, x='time', y='value', hue='class', fill=True, gap=0.1, saturation=0.5, palette='bright')
    plt.title('Box + Scatter over time')
    plt.savefig(os.path.join(output_dir, "Box + Scatter over time.png"))
    plt.close()

    fig1.seek(0)
    fig2.seek(0)
    fig3.seek(0)
    fig4.seek(0)


async def safe_send_json(websocket: WebSocket, data: dict) -> bool:
    try:
        await websocket.send_json(data)
        return True
    except WebSocketDisconnect:
        print("Client disconnected during send.")
    except Exception as e:
        print(f"Unexpected error while sending: {e}")
    return False


async def send_error_and_close(websocket: WebSocket, message: str) -> None:
    await safe_send_json(websocket, {"error": message})
    await websocket.close()


async def generate_chat_session_id() -> int:
    return int(str(uuid.uuid4().int)[:18]) 


async def generate_custom_identifiers(df, prefix_columns, column_data, endpoint):
    suffix_column_names = []
    # Step 1: Create suffix columns based on cut_off values
    for cond in column_data:
        if cond.cut_off is not None:
            col = cond.variable_name
            val = cond.cut_off
            if val == int(val):
                val = int(val)
            suffix_col = f"{col}_suffix"
            df[suffix_col] = np.where(df[col] < val, f"{col}_LT_{val}", f"{col}_GT_{val}")
            suffix_column_names.append(suffix_col)

    # Step 2: Avoid duplicate columns and exclude endpoint from prefix
    transformed_columns = {cond.variable_name for cond in column_data if cond.cut_off is not None}
    clean_prefix_columns = [
        col for col in prefix_columns
        if col not in transformed_columns and col != endpoint
    ]

    # Step 3: Create the final custom column (excluding endpoint)
    df["custom_column"] = df[clean_prefix_columns + suffix_column_names].astype(str).agg("_".join, axis=1)

    # Step 4: Only return endpoint and custom_column in the result
    df = df[[endpoint, "custom_column"]]
    df = df.dropna()
    return df


async def generate_posterior(data):
    observed_values = np.array(list(data.values()))

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=50, sigma=50)  # Centered around 50, very broad prior
        sigma = pm.HalfNormal("sigma", sigma=50)
        likelihood = pm.Normal("likelihood", mu=mu, sigma=sigma, observed=observed_values)
        trace = pm.sample(2000, return_inferencedata=True)

    az.plot_trace(trace)
    plt.close()

    posterior_mu = trace.posterior["mu"].values.flatten()[100:]

    ref_val = np.mean(posterior_mu)
    less_than_ref = np.sum(posterior_mu < ref_val) / len(posterior_mu)
    greater_than_ref = np.sum(posterior_mu > ref_val) / len(posterior_mu)

    hist_values, bin_edges = np.histogram(posterior_mu, bins=20)
    posterior_data = {
        "hist_values": hist_values.tolist(),
        "bin_edges": bin_edges.tolist(),
        "mean": float(np.mean(posterior_mu)),
        "median": float(np.median(posterior_mu)),
        "mode": float(bin_edges[np.argmax(hist_values)]),
        "less_than_ref_pct": round(less_than_ref * 100, 1),
        "greater_than_ref_pct": round(greater_than_ref * 100, 1)
    }

    # plt.figure()
    # pm.plot_posterior(posterior_mu, color="#87ceeb", point_estimate="mean", ref_val=np.mean(observed_values))
    # plt.savefig("posterior_plot.png")  # Replace this with your actual path
    # plt.show()
    # plt.close()
    return posterior_data


async def cal_avg(trial_req, filtered_dataframe, bayes_result,  week_dataframe: Optional[pd.DataFrame] = None, db: Session = Depends(get_db_session)):
    try:
        trial = db.exec(select(Trial).where(Trial.id == trial_req.trial_id, Trial.status == True, Trial.deleted == False)).first()
        redis_client = redis.Redis()
        column_types = filtered_dataframe.dtypes.astype(str).tolist()
        org_id = 1        # TODO: replace with actual user org_id when auth is implemented
        channel = f"trial_updates:{org_id}"
        message = None
        def publish_message(data):
            message = {
                "status": 201,
                "message": f'Stop the trial {trial.name}',
                "data": data
            }
            asyncio.create_task(redis_client.publish(channel, json.dumps(message)))
            print("Condition passed, message published")
            return message

        # Case 1: One continuous and one categorical variable
        if len(filtered_dataframe.columns) == 2 and ('float' in column_types or 'float64' in column_types or 'int' in column_types or 'Int64' in column_types or 'int64' in column_types) and 'object' in column_types:
            message = None
            numeric_col = filtered_dataframe.select_dtypes(include=['number']).columns[0]
            df_numeric_col = week_dataframe.select_dtypes(include=['number']).columns[0]
            df_category_col = week_dataframe.select_dtypes(include=['object']).columns[0]
                    
            groupby_avg = week_dataframe.groupby(df_category_col)[df_numeric_col].mean()
            groupby_avg_dict = groupby_avg.to_dict()

            if groupby_avg_dict:
                groupby_avg_mean = groupby_avg.mean()
                groupby_avg_dict["avg"] = groupby_avg_mean
            else:
                groupby_avg_mean = None
                groupby_avg_dict["avg"] = None  # or 0 or some default value

            weekly_avg_data = groupby_avg_dict
            avg = filtered_dataframe[numeric_col].mean()

            if "bayes_factors" in bayes_result:
                bayes_value = bayes_result["bayes_factors"]
            elif "corr_bayes_factors_data" in bayes_result:
                bayes_value = bayes_result["corr_bayes_factors_data"]
                bayes_value = min(item["BF10"] for item in bayes_value.values())
            else:
                bayes_value = bayes_result 
            
            posterior_data = await generate_posterior(weekly_avg_data)
            
            avg_result = {"overall": avg, "weekly": weekly_avg_data, "posterior_data": posterior_data}
            continuous_config = next((col for col in trial_req.endpoint_data.column_data or [] if col.type is not None and col.type.lower() == "continuous"), None)
            if continuous_config:
                avg_result["overall"] = {continuous_config.variable_name: avg}

            if bayes_value < 0.1:
                message = publish_message({"BF10": bayes_value, "avg": groupby_avg_mean})

            return {
                "values": avg_result,
                "message": message
            }

        # Case 2: Multiple categorical variables
        elif len(filtered_dataframe.columns) >= 2 and all(ctype == 'object' for ctype in column_types):
            if trial_req.endpoint_data.is_primary:
                continuous_config = next((col for col in trial_req.endpoint_data.column_data or [] if col.group == "Endpoint"), None) # endpoint
                group_by_config = next((col for col in trial_req.endpoint_data.column_data or [] if col.group == "Primary Grouping"), None)  # primary grouping
                coloumn = continuous_config.variable_name
                group_by_col = group_by_config.variable_name
            else:
                primary_endpoint = db.exec(select(TrialEndPoint).where(TrialEndPoint.trial_id == trial_req.trial_id, TrialEndPoint.is_primary == True, TrialEndPoint.status == True, TrialEndPoint.deleted == False)).first()
                endpoint = next((col for col in primary_endpoint.column_data or [] if col.get("group") == "Endpoint"), None)
                coloumn = endpoint["variable_name"]
                group_by_col = "custom_column"

            # byes_min_value = min(bayes_result.values())
            byes_min_value = min(item["BF10"] for item in bayes_result.values())

            # overall
            overall_value_counts = filtered_dataframe[coloumn].value_counts()
            overall_percentage_distribution = (overall_value_counts / overall_value_counts.sum()) * 100
            # weekly
            value_counts = week_dataframe[coloumn].value_counts()
            weekly_percentage_distribution = (value_counts / value_counts.sum()) * 100

            weekly_group = week_dataframe.groupby(group_by_col)[coloumn].value_counts(normalize=True) * 100
            weekly_distribution = weekly_group.unstack().fillna(0)

            weekly_flat = {
                            f"{region} vs {category}": round(pct, 2)
                            for region, row in weekly_distribution.iterrows()
                            for category, pct in row.items()
                        }
            posterior_data = await generate_posterior(weekly_flat)
            weekly_flat["avg"] = weekly_percentage_distribution.to_dict()
            if float(byes_min_value) < 0.01:
                message =  publish_message({"message": "User specified primary endpoint satisfied and BF indicates support for Fail"})
            return {
                "values": {
                    "overall": overall_percentage_distribution.to_dict(),
                    "weekly": weekly_flat,
                    "posterior_data": posterior_data
                    },
                "message": message
            }
        return None
    except Exception as e:
        traceback_data = traceback.format_exc()
        print(f"An error occurred in cal_avg: {e} and the traceback data is {traceback_data}. Please check.")


async def mean_calculation(df: pd.DataFrame, column_data: list, primary_col: str) -> dict:
    try:
        df = df.dropna()
        result = defaultdict(dict)
        secondary_columns = [col for col in column_data if col["variable_name"] != primary_col]

        for secondary in secondary_columns:
            col_name = secondary["variable_name"]
            col_type = secondary["type"].lower()

            if col_type == "category":
                # === Overall percentage distribution ===
                overall_counts = df[col_name].value_counts()
                overall_pct = (overall_counts / overall_counts.sum()) * 100

                # === Grouped percentage distribution ===
                group_pct = df.groupby(primary_col)[col_name].value_counts(normalize=True) * 100
                distribution = group_pct.unstack().fillna(0)

                for group in distribution.index:
                    result[group][f'{col_name} (%)'] = {
                        category: round(pct, 2) for category, pct in distribution.loc[group].items()
                    }

                result["overall"] = result.get("overall", {})
                result["overall"][f'{col_name} (%)'] = {
                    category: round(pct, 2) for category, pct in overall_pct.items()
                }

            elif col_type == "continuous":
                cutoff = secondary.get("cut_off")
                if cutoff is None:
                    continue

                # Calculate overall means for values > cutoff and < cutoff
                overall_gt_mean = df.loc[df[col_name] > cutoff, col_name].mean()
                overall_lt_mean = df.loc[df[col_name] < cutoff, col_name].mean()

                result["overall"] = result.get("overall", {})
                result["overall"][col_name] = {
                    f"GT_{cutoff}": round(overall_gt_mean, 2) if not pd.isna(overall_gt_mean) else None,
                    f"LT_{cutoff}": round(overall_lt_mean, 2) if not pd.isna(overall_lt_mean) else None
                }

                grouped = df.groupby(primary_col)[col_name]

                for group, series in grouped:
                    gt_values = series[series > cutoff]
                    lt_values = series[series < cutoff]

                    gt_mean = round(gt_values.mean(), 2) if not gt_values.empty else None
                    lt_mean = round(lt_values.mean(), 2) if not lt_values.empty else None

                    if group not in result:
                        result[group] = {}

                    if col_name not in result[group]:
                        result[group][col_name] = {}

                    result[group][col_name][f"GT_{cutoff}"] = gt_mean
                    result[group][col_name][f"LT_{cutoff}"] = lt_mean

        return dict(result)
    except Exception as e:
        traceback_data = traceback.format_exc()
        print(f"An error occurred in mean_calculation: {e} and the traceback data is {traceback_data}. Please check.")
        return {"status": "error", "message": str(e)}


# def safe_float(val):
#     try:
#         if val is None or isinstance(val, str):
#             return None
#         if np.isnan(val) or np.isinf(val):
#             return None
#         return float(val)
#     except Exception:
#         return None

# def format_result(result):
#     return {
#         "T": safe_float(result["T"].values[0]),
#         "dof": safe_float(result["dof"].values[0]),
#         "p-val": safe_float(result["p-val"].values[0]),
#         "CI95%": [safe_float(x) for x in result["CI95%"].values[0]],
#         "cohen-d": safe_float(result["cohen-d"].values[0]),
#         "power": safe_float(result["power"].values[0]),
#     }

# async def case1_non_inferiority_test(df: pd.DataFrame, group1: str, group2: str, alpha: float):
#     try:
#         # Identify one continuous and one categorical column
#         continuous_col = next(col for col in df.columns if df[col].dtype.name in ['float64', 'int64', 'Int64'])
#         categorical_col = next(col for col in df.columns if df[col].dtype.name == 'object')
        

#         # Filter only the relevant groups
#         df_sub = df[df[categorical_col].isin([group1, group2])]
#         if df_sub.empty:
#             return {"status": "error", "message": f"No data found for groups: {group1}, {group2}"}

#         # Group and compute mean and count
#         groups = df_sub.groupby(categorical_col)[continuous_col].agg(['mean', 'count'])
#         if group1 not in groups.index or group2 not in groups.index:
#             return {"status": "error", "message": f"Both groups must exist in the data."}

#         # Convert mean to proportion (assuming percentage input)
#         m1, n1 = groups.loc[group1, 'mean'] / 100, groups.loc[group1, 'count']
#         m2, n2 = groups.loc[group2, 'mean'] / 100, groups.loc[group2, 'count']

#         diff = m1 - m2
#         se = np.sqrt(m1 * (1 - m1) / n1 + m2 * (1 - m2) / n2)
#         z = norm.ppf(1 - alpha / 2)
#         ci_lower = diff - z * se
#         ci_upper = diff + z * se

#         return {
#             "status": "success",
#             "group_pair": f"{group1} vs {group2}",
#             "mean_difference": safe_float(diff),
#             "ci_lower": safe_float(ci_lower),
#             "ci_upper": safe_float(ci_upper),
#             "z_score": safe_float(z)
#         }

#     except Exception as e:
#         return {"status": "error", "message": str(e)}


# async def case2_non_inferiority_test(df: pd.DataFrame, group1: str, group2: str, alpha: float):
#     try:
#         cat_cols = [col for col in df.columns if df[col].dtype == 'object']
#         if len(cat_cols) < 2:
#             return {"status": "error", "message": "Expected at least two categorical columns."}

#         group_col, outcome_col = cat_cols[:2]

#         df['outcome_code'] = df[outcome_col].astype('category').cat.codes

#         df_group1 = df[df[group_col] == group1]
#         df_group2 = df[df[group_col] == group2]

#         mean1 = df_group1['outcome_code'].mean()
#         std1 = df_group1['outcome_code'].std()
#         n1 = df_group1.shape[0]

#         mean2 = df_group2['outcome_code'].mean()
#         std2 = df_group2['outcome_code'].std()
#         n2 = df_group2.shape[0]

#         diff = mean1 - mean2
#         se = np.sqrt((std1 ** 2 / n1) + (std2 ** 2 / n2))
#         z = norm.ppf(1 - alpha / 2)
#         ci_lower = diff - z * se
#         ci_upper = diff + z * se

#         return {
#             "status": "success",
#             "group_pair": f"{group1} vs {group2}",
#             "mean_difference": safe_float(diff),
#             "ci_lower": safe_float(ci_lower),
#             "ci_upper": safe_float(ci_upper),
#             "z_score": safe_float(z)
#         }

#     except Exception as e:
#         return {"status": "error", "message": str(e)}
    
# async def generate_ttest_result(df, alpha):
#     try:
#         dataframe_copy = df.copy()
#         continuous_col = next(col for col in df.columns if df[col].dtype.name in ['float64', 'int64', 'Int64'])
#         categorical_col = next(col for col in df.columns if df[col].dtype.name == 'object')

#         df[continuous_col] = pd.to_numeric(df[continuous_col], errors='coerce')
#         df[categorical_col] = df[categorical_col].astype('category')
#         groups = df.groupby(categorical_col)[continuous_col].apply(list)

#         if len(groups) < 2:
#             raise HTTPException(status_code=400, detail="At least 2 groups are required.")
#         if df[continuous_col].nunique() <= 1:
#             raise HTTPException(status_code=400, detail="All values in the continuous column are the same.")

#         group_dict = groups.to_dict()
#         pairwise_stats, non_inferiority_test = {}, {}
#         for g1, g2 in combinations(group_dict, 2):
#             result = pg.ttest(group_dict[g1], group_dict[g2], paired=False, alternative='two-sided')
#             ttest_result = format_result(result)
#             ni_test = {"status": "error", "ci_lower": None, "message": "Non-inferiority test not performed, Alpha is missing."}
#             if alpha:
#                 ni_test = await case1_non_inferiority_test(dataframe_copy, g1, g2, alpha)
#                 ttest_result["ci_lower"] = ni_test.get("ci_lower", None)
#                 ttest_result["ci_upper"] = ni_test.get("ci_upper", None)
#             pairwise_stats[f"{g1} vs {g2}"] = ttest_result
#             non_inferiority_test[f"{g1} vs {g2}"] = ni_test
#         return {
#             "pairwise_stats": pairwise_stats,
#             "non_inferiority_test": non_inferiority_test
#         }
#     except Exception as e:
#         return {"status": "error", "message": str(e)}
    

# async def generate_chi2_result(df, alpha):
#     try:
#         dataframe_copy = df.copy()
#         cat_cols = [col for col in df.columns if df[col].dtype.name == 'object']
#         if len(cat_cols) < 2:
#             return {"status": "error", "message": "At least two categorical columns are required."}

#         col1, col2 = cat_cols[:2]
#         pairwise_stats, non_inferiority_test, ni_test = {}, {}, {}
#         ci_lower = None

#         unique_groups = df[col1].unique()
#         for g1, g2 in combinations(unique_groups, 2):
#             df_sub = df[df[col1].isin([g1, g2])]
#             observations = pd.crosstab(df_sub[col1], df_sub[col2], margins=False)

#             if observations.shape[0] < 2 or observations.shape[1] < 2:
#                 continue

#             chi2, p, dof, expected = chi2_contingency(observations)

#             categories = sorted(df[col2].unique())
#             outcome_mapping = {cat: i for i, cat in enumerate(categories)}

#             group_values = {
#                 g1: df_sub[df_sub[col1] == g1][col2].map(outcome_mapping).dropna(),
#                 g2: df_sub[df_sub[col1] == g2][col2].map(outcome_mapping).dropna()
#             }

#             g1_vals = group_values[g1]
#             g2_vals = group_values[g2]
#             cohens_d = power = None

#             if len(g1_vals) >= 2 and len(g2_vals) >= 2:
#                 mean1, mean2 = g1_vals.mean(), g2_vals.mean()
#                 std1, std2 = g1_vals.std(ddof=1), g2_vals.std(ddof=1)
#                 n1, n2 = len(g1_vals), len(g2_vals)
#                 pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
#                 cohens_d = safe_float((mean1 - mean2) / pooled_std)

#                 try:
#                     analysis = TTestIndPower()
#                     power = safe_float(analysis.solve_power(effect_size=cohens_d, nobs1=n1, alpha=0.05, power=None, ratio=1.0))
#                 except:
#                     power = None
#             ni_test = {"status": "error", "ci_lower": None, "message": "Non-inferiority test not performed, Alpha is missing."}
#             if alpha:
#                 ni_test = await case2_non_inferiority_test(dataframe_copy, g1, g2, alpha)
#                 ci_lower = ni_test.get("ci_lower") if ni_test.get("status") == "success" else None
#                 ci_upper = ni_test.get("ci_upper") if ni_test.get("status") == "success" else None

#             pairwise_stats[f"{g1} vs {g2}"] = {
#                 "T": safe_float(chi2),
#                 "dof": int(dof),
#                 "p-val": safe_float(p),
#                 "CI95%": None,
#                 "cohen-d": cohens_d,
#                 "power": power,
#                 "ci_lower" : ci_lower,
#                 "ci_upper" : ci_upper
#             }
#             non_inferiority_test[f"{g1} vs {g2}"] = ni_test

#         return {
#             "pairwise_stats": pairwise_stats,
#             "non_inferiority_test": non_inferiority_test
#         }
    

#     except Exception as e:
#         return {"status": "error", "message": str(e)}


def normalize_column_data(column_data):
    return [col.dict() if hasattr(col, "dict") else col for col in column_data]

class TrialStatAnalyzer:
    def __init__(self, df: pd.DataFrame, alpha: float = None):
        self.df = df.copy()
        self.alpha = alpha
        self.column_types = self.df.dtypes.astype(str).values.tolist()

    def safe_float(self, val):
        try:
            if val is None or isinstance(val, str):
                return None
            if np.isnan(val) or np.isinf(val):
                return None
            return float(val)
        except Exception:
            return None

    def format_result(self, result):
        return {
            "T": self.safe_float(result["T"].values[0]),
            "dof": self.safe_float(result["dof"].values[0]),
            "p-val": self.safe_float(result["p-val"].values[0]),
            "CI95%": [self.safe_float(x) for x in result["CI95%"].values[0]],
            "cohen-d": self.safe_float(result["cohen-d"].values[0]),
            "power": self.safe_float(result["power"].values[0]),
        }

    async def run_ttest(self):
        try:
            continuous_col = next(col for col in self.df.columns if self.df[col].dtype.name in ['float64', 'int64', 'Int64'])
            categorical_col = next(col for col in self.df.columns if self.df[col].dtype.name == 'object')

            self.df[continuous_col] = pd.to_numeric(self.df[continuous_col], errors='coerce')
            self.df[categorical_col] = self.df[categorical_col].astype('category')
            groups = self.df.groupby(categorical_col, observed=True)[continuous_col].apply(list)

            if len(groups) < 2:
                return {"status": "error", "message": "At least 2 groups are required."}
            if self.df[continuous_col].nunique() <= 1:
                return {"status": "error", "message": "All values in the continuous column are the same."}

            group_dict = groups.to_dict()
            pairwise_stats, non_inferiority_test = {}, {}

            for g1, g2 in combinations(group_dict, 2):
                result = pg.ttest(group_dict[g1], group_dict[g2], paired=False, alternative='two-sided')
                ttest_result = self.format_result(result)
                ni_test = {"status": "error", "ci_lower": None, "message": "Alpha is missing."}
                if self.alpha:
                    ni_test = await self._case1_non_inferiority_test(g1, g2, continuous_col, categorical_col)
                    ttest_result["ci_lower"] = ni_test.get("ci_lower")
                    ttest_result["ci_upper"] = ni_test.get("ci_upper")
                pairwise_stats[f"{g1} vs {g2}"] = ttest_result
                non_inferiority_test[f"{g1} vs {g2}"] = ni_test

            return {
                "pairwise_stats": pairwise_stats,
                "non_inferiority_test": non_inferiority_test
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def run_chi2(self):
        try:
            cat_cols = [col for col in self.df.columns if self.df[col].dtype.name == 'object']
            col1, col2 = cat_cols[:2]
            unique_groups = self.df[col1].unique()
            pairwise_stats, non_inferiority_test = {}, {}

            for g1, g2 in combinations(unique_groups, 2):
                df_sub = self.df[self.df[col1].isin([g1, g2])]
                observations = pd.crosstab(df_sub[col1], df_sub[col2], margins=False)

                if observations.shape[0] < 2 or observations.shape[1] < 2:
                    continue

                chi2, p, dof, expected = chi2_contingency(observations)

                categories = sorted(df_sub[col2].unique())
                outcome_mapping = {cat: i for i, cat in enumerate(categories)}

                g1_vals = df_sub[df_sub[col1] == g1][col2].map(outcome_mapping).dropna()
                g2_vals = df_sub[df_sub[col1] == g2][col2].map(outcome_mapping).dropna()

                cohens_d = power = None
                if len(g1_vals) >= 2 and len(g2_vals) >= 2:
                    mean1, mean2 = g1_vals.mean(), g2_vals.mean()
                    std1, std2 = g1_vals.std(ddof=1), g2_vals.std(ddof=1)
                    n1, n2 = len(g1_vals), len(g2_vals)
                    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
                    cohens_d = self.safe_float((mean1 - mean2) / pooled_std)

                    try:
                        analysis = TTestIndPower()
                        power = self.safe_float(analysis.solve_power(effect_size=cohens_d, nobs1=n1, alpha=0.05))
                    except Exception:
                        power = None

                ni_test = {"status": "error", "ci_lower": None, "message": "Alpha is missing."}
                if self.alpha:
                    ni_test = await self._case2_non_inferiority_test(g1, g2, col1, col2)

                pairwise_stats[f"{g1} vs {g2}"] = {
                    "T": self.safe_float(chi2),
                    "dof": int(dof),
                    "p-val": self.safe_float(p),
                    "CI95%": None,
                    "cohen-d": cohens_d,
                    "power": power,
                    "ci_lower": ni_test.get("ci_lower"),
                    "ci_upper": ni_test.get("ci_upper")
                }
                non_inferiority_test[f"{g1} vs {g2}"] = ni_test

            return {
                "pairwise_stats": pairwise_stats,
                "non_inferiority_test": non_inferiority_test
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def _case1_non_inferiority_test(self, g1, g2, cont_col, cat_col):
        try:
            df_sub = self.df[self.df[cat_col].isin([g1, g2])]
            groups = df_sub.groupby(cat_col, observed=True)[cont_col].agg(['mean', 'count'])

            m1, n1 = groups.loc[g1, 'mean'] / 100, groups.loc[g1, 'count']
            m2, n2 = groups.loc[g2, 'mean'] / 100, groups.loc[g2, 'count']

            diff = m1 - m2
            se = np.sqrt(m1 * (1 - m1) / n1 + m2 * (1 - m2) / n2)
            z = norm.ppf(1 - self.alpha / 2)
            ci_lower = diff - z * se
            ci_upper = diff + z * se

            return {
                "status": "success",
                "group_pair": f"{g1} vs {g2}",
                "mean_difference": self.safe_float(diff),
                "ci_lower": self.safe_float(ci_lower),
                "ci_upper": self.safe_float(ci_upper),
                "z_score": self.safe_float(z)
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def _case2_non_inferiority_test(self, g1, g2, group_col, outcome_col):
        try:
            self.df['outcome_code'] = self.df[outcome_col].astype('category').cat.codes
            df1 = self.df[self.df[group_col] == g1]
            df2 = self.df[self.df[group_col] == g2]

            mean1, std1, n1 = df1['outcome_code'].mean(), df1['outcome_code'].std(), len(df1)
            mean2, std2, n2 = df2['outcome_code'].mean(), df2['outcome_code'].std(), len(df2)

            diff = mean1 - mean2
            se = np.sqrt((std1 ** 2 / n1) + (std2 ** 2 / n2))
            z = norm.ppf(1 - self.alpha / 2)
            ci_lower = diff - z * se
            ci_upper = diff + z * se

            return {
                "status": "success",
                "group_pair": f"{g1} vs {g2}",
                "mean_difference": self.safe_float(diff),
                "ci_lower": self.safe_float(ci_lower),
                "ci_upper": self.safe_float(ci_upper),
                "z_score": self.safe_float(z)
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
        

async def plot_spearman_heatmap_with_pvalues(df: pd.DataFrame, title: str = 'Correlation Matrix with P-values', output_path: str = "correlation_heatmap.png") -> dict:
    try:
        df = df.dropna()
        int_cols = df.select_dtypes(include=['int64', 'Int64', 'float64', 'float32', 'Int32', 'int', "float"])
        if int_cols.empty:
            return {"error": "No numeric columns found."}

        # Compute Spearman correlation matrix
        corr_matrix = int_cols.corr(method='spearman')
        p_values = pd.DataFrame(index=corr_matrix.index, columns=corr_matrix.columns)

        for col1 in corr_matrix.columns:
            for col2 in corr_matrix.columns:
                if col1 != col2:
                    correlation, p_value = spearmanr(int_cols[col1], int_cols[col2])
                    p_values.loc[col1, col2] = p_value
                else:
                    p_values.loc[col1, col2] = None

        # annot_matrix = corr_matrix.copy()   # it cause deprication warning for flot to string conversion so update below
        annot_matrix = pd.DataFrame("", index=corr_matrix.index, columns=corr_matrix.columns, dtype=object)

        for col1 in corr_matrix.columns:
            for col2 in corr_matrix.columns:
                if col1 != col2:
                    p_val = p_values.loc[col1, col2]
                    annot_matrix.loc[col1, col2] = f"{corr_matrix.loc[col1, col2]:.2f}\np={float(p_val):.3f}" if pd.notnull(p_val) else ""
                else:
                    annot_matrix.loc[col1, col2] = ""

        # Optionally save plot (commented out for async use)
        # sns.heatmap(corr_matrix, annot=annot_matrix, fmt='', cmap='coolwarm', annot_kws={"size": 8})
        # plt.title(title)
        # plt.tight_layout()
        # plt.savefig(output_path)
        # plt.close()

        # Clean up values to be JSONB-compatible
        def clean_json_dict(d):
            return {
                k: {
                    sub_k: (
                        None if pd.isna(sub_v) or np.isinf(sub_v) else round(float(sub_v), 4)
                    )
                    for sub_k, sub_v in v.items()
                }
                for k, v in d.items()
            }

        corr_json = clean_json_dict(corr_matrix.to_dict())
        pval_json = clean_json_dict(p_values.to_dict())

        return {
            "correlation_matrix": corr_json,
            "p_values": pval_json,
            "plot_path": output_path
        }

    except Exception as e:
        traceback_data = traceback.format_exc()
        print(f"An error occurred in plot_spearman_heatmap_with_pvalues: {e} and the traceback data is {traceback_data}. Please check.")
        return {"status": "error", "message": str(e)}
   

class GenerateKMPlots:
    def __init__(self):
        pass

    @staticmethod
    def normalize_column_data(column_data):
        return [col.dict() if hasattr(col, "dict") else col for col in column_data]

    async def generate_event_observed(self, data, primary_col, column_data):
        try:
            df = data.copy()
            df["date"] = pd.to_datetime(df["date"], dayfirst=True)
            min_date = df["date"].min()
            df["duration"] = (df["date"] - min_date).dt.days
            column_data = self.normalize_column_data(column_data)

            endpoint_configs = [
                col for col in column_data
                if col.get("cut_off") is not None or col.get("category_value") is not None
            ]

            if not endpoint_configs:
                raise ValueError("No valid columns with 'cut_off' or 'category_value' found.")

            def row_event_observed(row):
                for config in endpoint_configs:
                    variable = config["variable_name"]
                    cut_off = config.get("cut_off")
                    category_value = config.get("category_value")
                    value = row.get(variable)

                    if cut_off is not None:
                        try:
                            if float(value) > float(cut_off):
                                return 0
                        except (ValueError, TypeError):
                            return 0
                    elif category_value is not None and value != category_value:
                        return 0
                return 1

            df["event_observed"] = df.apply(row_event_observed, axis=1)

            event_observed_by_group = {}
            durations_by_group = {}
            for group, group_df in df.groupby(primary_col):
                event_observed_by_group[f"EventObserved_{group}"] = group_df["event_observed"].tolist()
                durations_by_group[f"Durations_{group}"] = group_df["duration"].tolist()

            return await self.km_plot_and_logrank(durations_by_group, event_observed_by_group)
        except Exception as e:
            traceback_data = traceback.format_exc()
            print(f"Error in generate_event_observed: {e}\nTraceback: {traceback_data}")
            return {"status": "error", "message": str(e)}

    async def km_plot_and_logrank(self, durations_by_group, event_observed_dict):
        try:
            kmf = KaplanMeierFitter()
            km_curves = {}
            results = []

            for group_name, events in event_observed_dict.items():
                label = group_name.replace("EventObserved_", "")
                durations = durations_by_group[f"Durations_{label}"]
                if len(durations) != len(events):
                    raise ValueError(f"Length mismatch for group '{label}'")

                kmf.fit(durations, events, label=label)
                survival_df = kmf.survival_function_.reset_index()
                survival_df.columns = ["time", "survival_probability"]
                km_curves[label] = survival_df.to_dict(orient="records")

            group_items = list(event_observed_dict.items())
            for (name1, events1), (name2, events2) in combinations(group_items, 2):
                label1 = name1.replace("EventObserved_", "")
                label2 = name2.replace("EventObserved_", "")
                durations1 = durations_by_group[f"Durations_{label1}"]
                durations2 = durations_by_group[f"Durations_{label2}"]

                test = logrank_test(durations1, durations2, events1, events2)
                results.append({
                    "group1": label1,
                    "group2": label2,
                    "group": f"{label1} vs {label2}",
                    "p_value": test.p_value,
                    "test_statistic": test.test_statistic
                })

            return {
                "logrank_results": results,
                "km_curves": km_curves
            }
        except Exception as e:
            traceback_data = traceback.format_exc()
            print(f"Error in km_plot_and_logrank: {e}\nTraceback: {traceback_data}")
            return {"status": "error", "message": str(e)}
        

class StandardScalerAndBARTClassifier:
    def __init__(self, df, group_col, endpoint_col, endpoint_config):
        self.df = df.copy()
        self.group_col = group_col
        self.endpoint_col = endpoint_col
        self.outcome_col = 'outcome'
        self.endpoint_config = SimpleNamespace(**endpoint_config) if isinstance(endpoint_config, dict) else endpoint_config
        self.model = None
        self.scaler = None
        self.features = []
        self.display_features = []
        self.X_scaled = None
        self.y = None
        self.X = None
        self.category_mappings = {}
        self.trace = None
        self.μ = None


    def convert_df_to_int(self):
        def row_event_observed(row):
            variable = getattr(self.endpoint_config, "variable_name", None)
            cut_off = getattr(self.endpoint_config, "cut_off", None)
            category_value = getattr(self.endpoint_config, "category_value", None)

            value = row.get(variable)
            if cut_off is not None:
                try:
                    return int(float(value) > float(cut_off)) ^ 1
                except (ValueError, TypeError):
                    return 0
            elif category_value is not None:
                return int(value == category_value)
            else:
                raise ValueError(f"Invalid config for variable '{variable}': Must have 'cut_off' or 'category_value'.")

        self.df[self.outcome_col] = self.df.apply(row_event_observed, axis=1)

        for col in self.df.columns:
            if col in {self.group_col, self.outcome_col, "date"}:
                continue
            if self.df[col].dtype == object or self.df[col].dtype.name == 'category':
                self.df[col], uniques = pd.factorize(self.df[col])
                self.category_mappings[col] = dict(enumerate(uniques))
            else:
                self.df[col] = self.df[col].astype(int)

    def preprocess(self):
        self.convert_df_to_int()
        self.df = self.df.dropna()
        self.features = [col for col in self.df.columns if col not in [self.group_col, self.outcome_col, 'date']]
        self.display_features = [col for col in self.features if col != self.endpoint_col]
        
        self.X = self.df[self.features]
        self.y = self.df[self.outcome_col]
        self.scaler = StandardScaler()
        print("self.scaler", self.scaler)
        
        self.X_scaled = self.scaler.fit_transform(self.X)

    def train_model(self):
        with pm.Model() as model:
            bart = pmb.BART("μ", X=self.X_scaled, Y=self.y, m=50)
            self.μ = bart
            y_obs = pm.Bernoulli("y_obs", logit_p=self.μ, observed=self.y)
            step = pmb.PGBART()
            self.trace = pm.sample(draws=200, tune=100, chains=2, step=step)
            # self.trace = pm.sample(draws=1000, tune=500, chains=2, step=step, random_seed=42)
            self.model = model
        try:
            if "μ" in self.trace.posterior:
                inclusion_probs = self.trace.sample_stats.get("variable_inclusion", None)
                if inclusion_probs is not None:
                    inclusion_probs_mean = inclusion_probs.mean(dim=["chain", "draw"]).values
                    self.feature_importance = dict(zip(self.features, inclusion_probs_mean))
                else:
                    self.feature_importance = dict(zip(self.features, [np.nan] * len(self.features)))
            else:
                self.feature_importance = dict(zip(self.features, [np.nan] * len(self.features)))
        except Exception as e:
            print(f"Error computing inclusion probabilities: {e}")
            self.feature_importance = dict(zip(self.features, [np.nan] * len(self.features)))

    @staticmethod
    def format_text_block(title, block):
        html = f"<h3>{title}</h3>"
        for line in block.strip().split("\n"):
            if line.startswith("🔹"):
                html += f"<p><strong>{line}</strong></p>"
            elif line.startswith("⚠️"):
                html += f"<p style='color:red'>{line}</p>"
            elif line.strip():
                html += f"<p>{line}</p>"
        return html

    def _generate_html_summary(self, importances, group_diff, outcome_html, sample_issues, recommendations):
        importance_html = "<h3>🔍 Top Feature Importances</h3><table border='1'><tr><th>Feature</th><th>Importance</th></tr>"
        for feat, val in importances.head(10).items():
            importance_html += f"<tr><td>{feat}</td><td>{val:.4f}</td></tr>"
        importance_html += "</table>"

        group_diff_html = group_diff
        outcome_html_section = f"""<div class='outcome-section'><h3>📈 Outcome Rate by Subgroup</h3>""" + outcome_html + "</div>"
        sample_html = self.format_text_block("⚠️ Sample Size Warnings", sample_issues)

        recommendation_html = "<h3>💡 Recommendations</h3><ul>"
        for rec in recommendations:
            recommendation_html += f"<li>{rec}</li>"
        recommendation_html += "</ul>"

        return importance_html + group_diff_html + outcome_html_section + sample_html + recommendation_html

    def _group_difference_insights(self):
        html = ['<h3>📊 Group-Level Variable Comparison</h3>']
        groups = sorted(self.df[self.group_col].dropna().unique())

        for col in self.display_features:
            try:
                if self.df[col].dtype == 'object' or self.df[col].nunique() <= 5:
                    counts = self.df.groupby(self.group_col)[col].value_counts(normalize=True).unstack().fillna(0)
                    mapping = self.category_mappings.get(col, {})
                    html.append(f"<h4>🔹 {col} Distribution</h4>")
                    html.append("<table border='1'><tr><th>Category</th>" +
                                "".join(f"<th>{grp}</th>" for grp in groups) +
                                "<th>Max Δ (%)</th><th>Note</th></tr>")
                    for cat in counts.columns:
                        cat_label = mapping.get(cat, str(cat))
                        row = [f"<td>{cat_label}</td>"]
                        for grp in groups:
                            pct = counts.loc[grp, cat] * 100 if grp in counts.index else 0
                            row.append(f"<td>{pct:.1f}%</td>")
                        deltas = [abs(counts.loc[grp1, cat] - counts.loc[grp2, cat]) * 100 
                                for i, grp1 in enumerate(groups) for grp2 in groups[i+1:] 
                                if grp1 in counts.index and grp2 in counts.index]
                        max_delta = max(deltas) if deltas else 0
                        note = "⚠️" if max_delta > 10 else ""
                        row.append(f"<td>{max_delta:.1f}%</td><td>{note}</td>")
                        html.append("<tr>" + "".join(row) + "</tr>")
                    html.append("</table>")

                else:
                    means = self.df.groupby(self.group_col)[col].mean()
                    html.append(f"<h4>🔹 {col} (Mean)</h4>")
                    html.append("<table border='1'><tr><th>Group</th><th>Mean</th></tr>")
                    for grp in groups:
                        val = means.get(grp, 0)
                        html.append(f"<tr><td>{grp}</td><td>{val:.2f}</td></tr>")
                    html.append("</table>")

                    deltas = [abs(means.get(grp1, 0) - means.get(grp2, 0)) 
                            for i, grp1 in enumerate(groups) for grp2 in groups[i+1:]]
                    max_delta = max(deltas) if deltas else 0
                    if deltas:
                        html.append(f"<p><strong>Max Δ = {max_delta:.2f}</strong></p>")
                    if max_delta > 0.5:
                        html.append("<p style='color:red;'>⚠️ Consider rebalancing or stratifying on this variable.</p>")
            except Exception:
                continue
        return "".join(html)

    def _outcome_by_subgroup(self):
        html_sections = []
        categorical_cols = [col for col in self.display_features if self.df[col].nunique() <= 10]

        for col in categorical_cols:
            try:
                rate = self.df.groupby([self.group_col, col])[self.outcome_col].mean().unstack()
                rate = rate.round(2).fillna('-')

                mapping = self.category_mappings.get(col)
                if mapping:
                    rate.columns = [mapping.get(c, c) for c in rate.columns]

                rate.index.name = self.group_col
                rate.columns.name = None

                html_table = rate.to_html(border=1, classes="dataframe outcome-table", na_rep='-', escape=False).replace('\n', '')
                html_sections.append(f"<h4>🔹 Outcome rates by <i>{col}</i>:</h4>")
                html_sections.append(html_table)

            except Exception as e:
                print(f"Error processing column {col}: {e}")
                continue

        return "".join(html_sections)

    def _sample_size_warning(self, threshold=0.1):
        warnings = []
        categorical_cols = [col for col in self.display_features if self.df[col].nunique() <= 10]
        for col in categorical_cols:
            value_counts = self.df[col].value_counts(normalize=True)
            for val, pct in value_counts.items():
                if pct < threshold:
                    val_label = self.category_mappings.get(col, {}).get(val, val)
                    warnings.append(f"⚠️ Subgroup {col} = {val_label} represents only {pct*100:.1f}% of the population.")
        return "\n".join(warnings)

    def generate_recommendations(self, importances, sample_size_text, all_importances):
        recommendations = []

        if any(abs(val) > 0.5 for val in all_importances.dropna()):
            recommendations.append("⚠️ Consider rebalancing or stratifying the dataset based on observed group differences.")

        if sample_size_text:
            recommendations.append("⚠️ Collect more data for underrepresented subgroups to improve model robustness.")

        if self.endpoint_col in all_importances and all_importances[self.endpoint_col] > 0.2:
            recommendations.append(f"⚠️ The endpoint column '{self.endpoint_col}' is highly predictive, consider potential data leakage.")

        if importances.empty:
            recommendations.append("⚠️ No feature importances were computed. Check if the input data has enough variation or valid columns.")
        elif importances.head(1).values[0] < 0.01:
            recommendations.append("⚠️ No strong predictors found. Consider feature engineering or revisiting variable selection.")

        return recommendations

    def generate_insights(self):
        all_importances = pd.Series(self.feature_importance, index=self.features)
        importances = all_importances[all_importances.index != self.endpoint_col].sort_values(ascending=False)
        group_diff_text = self._group_difference_insights()
        outcome_text = self._outcome_by_subgroup()
        sample_size_text = self._sample_size_warning()
        recommendations = self.generate_recommendations(importances, sample_size_text, all_importances)

        html_summary = self._generate_html_summary(importances, group_diff_text, outcome_text, sample_size_text, recommendations)
        print("html_summary", html_summary)
        return html_summary

    @classmethod
    async def run(cls, df: pd.DataFrame, group_col: str, endpoint_col: str, endpoint_config) -> dict:
        instance = cls(df, group_col, endpoint_col, endpoint_config)
        instance.preprocess()
        instance.train_model()
        return instance.generate_insights()


class StandardScalerAndRandomForestClassifier:
    def __init__(self, df, group_col, endpoint_col, endpoint_config):
        self.df = df.copy()
        self.group_col = group_col
        self.endpoint_col = endpoint_col
        self.outcome_col = 'outcome'
        self.endpoint_config = SimpleNamespace(**endpoint_config) if isinstance(endpoint_config, dict) else endpoint_config
        self.model = None
        self.scaler = None
        self.features = []
        self.display_features = []
        self.X_scaled = None
        self.y = None
        self.X = None
        self.category_mappings = {}

    def convert_df_to_int(self):
        def row_event_observed(row):
            variable = getattr(self.endpoint_config, "variable_name", None)
            cut_off = getattr(self.endpoint_config, "cut_off", None)
            category_value = getattr(self.endpoint_config, "category_value", None)

            value = row.get(variable)
            if cut_off is not None:
                try:
                    return int(float(value) > float(cut_off)) ^ 1
                except (ValueError, TypeError):
                    return 0
            elif category_value is not None:
                return int(value == category_value)
            else:
                raise ValueError(f"Invalid config for variable '{variable}': Must have 'cut_off' or 'category_value'.")

        self.df[self.outcome_col] = self.df.apply(row_event_observed, axis=1)

        for col in self.df.columns:
            if col in {self.group_col, self.outcome_col, "date"}:
                continue
            if self.df[col].dtype == object or self.df[col].dtype.name == 'category':
                self.df[col], uniques = pd.factorize(self.df[col])
                self.category_mappings[col] = dict(enumerate(uniques))
            else:
                self.df[col] = self.df[col].astype(int)

    def preprocess(self):
        self.convert_df_to_int()
        self.df = self.df.dropna()
        self.features = [col for col in self.df.columns if col not in [self.group_col, self.outcome_col, 'date']]
        self.display_features = [col for col in self.features if col != self.endpoint_col]
        
        self.X = self.df[self.features]
        self.y = self.df[self.outcome_col]
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)

    def train_model(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(self.X_scaled, self.y)
        self.feature_importance = dict(zip(self.features, self.model.feature_importances_))

    @staticmethod
    def format_text_block(title, block):
        html = f"<h3>{title}</h3>"
        for line in block.strip().split("\n"):
            if line.startswith("🔹"):
                html += f"<p><strong>{line}</strong></p>"
            elif line.startswith("⚠️"):
                html += f"<p style='color:red'>{line}</p>"
            elif line.strip():
                html += f"<p>{line}</p>"
        return html

    def _generate_html_summary(self, importances, group_diff, outcome_html, sample_issues, recommendations):
        importance_html = "<h3>🔍 Top Feature Importances</h3><table border='1'><tr><th>Feature</th><th>Importance</th></tr>"
        for feat, val in importances.head(10).items():
            importance_html += f"<tr><td>{feat}</td><td>{val:.4f}</td></tr>"
        importance_html += "</table>"

        group_diff_html = group_diff
        outcome_html_section = f"""<div class='outcome-section'><h3>📈 Outcome Rate by Subgroup</h3>""" + outcome_html + "</div>"
        sample_html = self.format_text_block("⚠️ Sample Size Warnings", sample_issues)

        recommendation_html = "<h3>💡 Recommendations</h3><ul>"
        for rec in recommendations:
            recommendation_html += f"<li>{rec}</li>"
        recommendation_html += "</ul>"

        return importance_html + group_diff_html + outcome_html_section + sample_html + recommendation_html

    def _group_difference_insights(self):
        html = ['<h3>📊 Group-Level Variable Comparison</h3>']
        groups = sorted(self.df[self.group_col].dropna().unique())

        for col in self.display_features:
            try:
                if self.df[col].dtype == 'object' or self.df[col].nunique() <= 5:
                    counts = self.df.groupby(self.group_col)[col].value_counts(normalize=True).unstack().fillna(0)
                    mapping = self.category_mappings.get(col, {})
                    html.append(f"<h4>🔹 {col} Distribution</h4>")
                    html.append("<table border='1'><tr><th>Category</th>" +
                                "".join(f"<th>{grp}</th>" for grp in groups) +
                                "<th>Max Δ (%)</th><th>Note</th></tr>")
                    for cat in counts.columns:
                        cat_label = mapping.get(cat, str(cat))
                        row = [f"<td>{cat_label}</td>"]
                        for grp in groups:
                            pct = counts.loc[grp, cat] * 100 if grp in counts.index else 0
                            row.append(f"<td>{pct:.1f}%</td>")
                        deltas = [abs(counts.loc[grp1, cat] - counts.loc[grp2, cat]) * 100 
                                for i, grp1 in enumerate(groups) for grp2 in groups[i+1:] 
                                if grp1 in counts.index and grp2 in counts.index]
                        max_delta = max(deltas) if deltas else 0
                        note = "⚠️" if max_delta > 10 else ""
                        row.append(f"<td>{max_delta:.1f}%</td><td>{note}</td>")
                        html.append("<tr>" + "".join(row) + "</tr>")
                    html.append("</table>")

                else:
                    means = self.df.groupby(self.group_col)[col].mean()
                    html.append(f"<h4>🔹 {col} (Mean)</h4>")
                    html.append("<table border='1'><tr><th>Group</th><th>Mean</th></tr>")
                    for grp in groups:
                        val = means.get(grp, 0)
                        html.append(f"<tr><td>{grp}</td><td>{val:.2f}</td></tr>")
                    html.append("</table>")

                    deltas = [abs(means.get(grp1, 0) - means.get(grp2, 0)) 
                            for i, grp1 in enumerate(groups) for grp2 in groups[i+1:]]
                    max_delta = max(deltas) if deltas else 0
                    if deltas:
                        html.append(f"<p><strong>Max Δ = {max_delta:.2f}</strong></p>")
                    if max_delta > 0.5:
                        html.append("<p style='color:red;'>⚠️ Consider rebalancing or stratifying on this variable.</p>")
            except Exception:
                continue
        return "".join(html)

    def _outcome_by_subgroup(self):
        html_sections = []
        categorical_cols = [col for col in self.display_features if self.df[col].nunique() <= 10]

        for col in categorical_cols:
            try:
                rate = self.df.groupby([self.group_col, col])[self.outcome_col].mean().unstack()
                rate = rate.round(2).fillna('-')

                mapping = self.category_mappings.get(col)
                if mapping:
                    rate.columns = [mapping.get(c, c) for c in rate.columns]

                rate.index.name = self.group_col
                rate.columns.name = None

                html_table = rate.to_html(border=1, classes="dataframe outcome-table", na_rep='-', escape=False).replace('\n', '')
                html_sections.append(f"<h4>🔹 Outcome rates by <i>{col}</i>:</h4>")
                html_sections.append(html_table)

            except Exception as e:
                print(f"Error processing column {col}: {e}")
                continue

        return "".join(html_sections)

    def _sample_size_warning(self, threshold=0.1):
        warnings = []
        categorical_cols = [col for col in self.display_features if self.df[col].nunique() <= 10]
        for col in categorical_cols:
            value_counts = self.df[col].value_counts(normalize=True)
            for val, pct in value_counts.items():
                if pct < threshold:
                    val_label = self.category_mappings.get(col, {}).get(val, val)
                    warnings.append(f"⚠️ Subgroup {col} = {val_label} represents only {pct*100:.1f}% of the population.")
        return "\n".join(warnings)

    def generate_recommendations(self, importances, sample_size_text, all_importances):
        recommendations = []

        if any(abs(val) > 0.5 for val in all_importances.dropna()):
            recommendations.append("⚠️ Consider rebalancing or stratifying the dataset based on observed group differences.")

        if sample_size_text:
            recommendations.append("⚠️ Collect more data for underrepresented subgroups to improve model robustness.")

        if self.endpoint_col in all_importances and all_importances[self.endpoint_col] > 0.2:
            recommendations.append(f"⚠️ The endpoint column '{self.endpoint_col}' is highly predictive, consider potential data leakage.")

        if importances.empty:
            recommendations.append("⚠️ No feature importances were computed. Check if the input data has enough variation or valid columns.")
        elif importances.head(1).values[0] < 0.01:
            recommendations.append("⚠️ No strong predictors found. Consider feature engineering or revisiting variable selection.")

        return recommendations

    def generate_insights(self):
        all_importances = pd.Series(self.model.feature_importances_, index=self.features)
        importances = all_importances[all_importances.index != self.endpoint_col].sort_values(ascending=False)
        top_features_dict = importances.head(10).to_dict()

        group_diff_text = self._group_difference_insights()
        outcome_text = self._outcome_by_subgroup()
        sample_size_text = self._sample_size_warning()

        recommendations = self.generate_recommendations(importances, sample_size_text, all_importances)

        summary_text = "\n".join([
            "🔍 Top Feature Importances:\n" + importances.head(10).to_string(),
            group_diff_text,
            outcome_text,
            sample_size_text,
            "\n".join(recommendations)
        ])

        html_summary = self._generate_html_summary(importances, group_diff_text, outcome_text, sample_size_text, recommendations)
        print("html_summary", html_summary)
        return html_summary

    @classmethod
    async def run(cls, df: pd.DataFrame, group_col: str, endpoint_col: str, endpoint_config) -> dict:
        instance = cls(df, group_col, endpoint_col, endpoint_config)
        instance.preprocess()
        instance.train_model()
        return instance.generate_insights()

