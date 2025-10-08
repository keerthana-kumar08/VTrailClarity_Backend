import pandas as pd
from typing import Optional
from datetime import datetime
from redis.asyncio import Redis
from redis.client import PubSub
import xml.etree.ElementTree as ET
from sqlalchemy.orm import selectinload
import json, io, ollama, traceback, asyncio
from sqlmodel import Session, select, func, and_, or_, update, delete

# from core.database import get_db_session
# from apis.v1.schemas import TrialCreate, TrialResponse, TrialPlotRequest
# from models.core import Trial, TrialData, TrialFile, Organisation

# from utils.genrate_graph import  case_1_graph, case_3_graph, case_4_graph
# from fastapi import APIRouter, Depends, Query, status, Request, HTTPException, UploadFile, File, Form
# from utils.utils import create_trial_data, create_trial_file, parse_xml, parse_json, replace_nan_with_none
# from utils.genrate_plot import case_1_continuous_categorical, case_2_multiple_categorical, case_3_two_continuous, case_4_multiple_categorical_continuous, case_5_discontinuous_crosstab

from ...core.database import get_db_session
from ...core.depends import get_redis_client
from .schemas import TrialCreate, TrialResponse, TrialPlotRequest, TrialTypeResponse, TrialDesignResponse, TrialChatHistoryResponse, TrialEndPointCreate, TrialEndPointResponse, TrialUniqueValueRequest
from ...models.core import Trial, TrialData, TrialFile, Organisation, TrialType, TrialDesign, TrialResult, TrialResultChatHistory, TrialEndPoint

from ...utils.genrate_graph import case_1_graph, case_3_graph, case_4_graph
from fastapi import (
    APIRouter,
    Depends,
    Query,
    status,
    Request,
    HTTPException,
    UploadFile,
    File,
    Form,
    WebSocket,
    WebSocketDisconnect,
)
from ...utils.utils import (
    create_trial_data,
    create_trial_file,
    parse_xml,
    parse_json,
    replace_nan_with_none,
    get_trial_rec,
    validate_entity,
    create_trial_result,
    safe_send_json,
    send_error_and_close,
    generate_chat_session_id,
    generate_weekly_result,
    get_custome_trial_rec,
    cal_avg,
    get_trial_rec_without_date_filter,
    generate_custom_identifiers,
    # generate_ttest_result,
    # case1_non_inferiority_test,
    # case2_non_inferiority_test,
    # generate_chi2_result,
    normalize_column_data,
    StandardScalerAndRandomForestClassifier,
    StandardScalerAndBARTClassifier,
    GenerateKMPlots,
    TrialStatAnalyzer,
    # scaled_rf_model,
    plot_spearman_heatmap_with_pvalues,
    mean_calculation
)
from ...utils.chat_buffer import (
    buffer_message_to_redis,
    flush_messages_from_redis_to_db,
    load_chat_history_from_db_to_redis,
    get_chat_history_from_redis,
    stream_chat_history_from_redis,
    flush_messages_from_redis,
)
from ...utils.genrate_plot import (
    case_1_continuous_categorical,
    case_2_multiple_categorical,
    case_3_two_continuous,
    case_4_multiple_categorical_continuous,
    case_5_discontinuous_crosstab,
)

router = APIRouter()


@router.post("/save_trial/", status_code=status.HTTP_201_CREATED)
async def save_trial(trial_data: TrialCreate, db: Session = Depends(get_db_session)):
    await validate_entity(db, Organisation, trial_data.organisation_id, "Organisation not found", 404)
    await validate_entity(db, TrialDesign, trial_data.design_id, "Invalid design_id")
    await validate_entity(db, TrialType, trial_data.type_id, "Invalid type_id")

    if trial_data.id:  
        trial = db.get(Trial, trial_data.id)  
        if not trial:
            raise HTTPException(status_code=404, detail="Trial not found")  

        for key, value in trial_data.model_dump(exclude_unset=True).items():
            setattr(trial, key, value)
    else:
        trial = Trial(**trial_data.model_dump())  
        db.add(trial) 
    db.commit()
    db.refresh(trial)
    return {"status": 200, "message": "Success", "data": {"trial_id": trial.id}}


@router.post("/trial-end-points/", status_code=status.HTTP_201_CREATED)
async def save_end_points(endpoint: TrialEndPointCreate, db: Session = Depends(get_db_session)):
    await validate_entity(db, Trial, endpoint.trial_id, "Invalid trial_id")
    endpoint_data = endpoint.model_dump(exclude_unset=True)

    if endpoint.id:
        db.exec(
            delete(TrialResult).where(
                TrialResult.endpoint_id == endpoint.id,
                TrialResult.is_system_generated == True
            )
        )
        db.commit()
        trial_endpoint = db.get(TrialEndPoint, endpoint.id)
        if not trial_endpoint:
            raise HTTPException(status_code=404, detail="Trial end point not found")

        for key, value in endpoint_data.items():
            setattr(trial_endpoint, key, value)
    else:
        trial_endpoint = TrialEndPoint(**endpoint_data)
        db.add(trial_endpoint)

    db.commit()
    db.refresh(trial_endpoint)

    await generate_weekly_result(trial_endpoint.trial_id, trial_endpoint, db)
    return {
        "status": 200,
        "message": "Success",
        "data": {"trial_endpoint_id": trial_endpoint.id}
    }


@router.delete("/delete_trial/{trial_id}", status_code=status.HTTP_200_OK)
async def delete_trial(trial_id: int, db: Session = Depends(get_db_session)):
    trial = db.get(Trial, trial_id)
    if not trial:
        raise HTTPException(status_code=404, detail="Trial not found")
    
    # Mark trial and related records as deleted
    trial.status, trial.deleted = False, True
    
    trial_files = db.exec(select(TrialFile).where(TrialFile.trial_id == trial_id)).all()
    trial_file_ids = [file.id for file in trial_files]
    
    db.exec(update(TrialFile).where(TrialFile.trial_id == trial_id).values(status=False, deleted=True))
    db.exec(update(TrialData).where(TrialData.trial_file_id.in_(trial_file_ids)).values(status=False, deleted=True))
    
    db.commit()
    return {"status": 200, "message": "Trial and associated records marked as deleted"}


@router.delete("/delete_endpoint/{endpoint_id}", status_code=status.HTTP_200_OK)
async def delete_endpoint(endpoint_id: int, db: Session = Depends(get_db_session)):
    endpoint = db.get(TrialEndPoint, endpoint_id)
    if not endpoint:
        raise HTTPException(status_code=404, detail="endpoint not found")
    endpoint.status, endpoint.deleted = False, True
    db.exec(update(TrialResult).where(TrialResult.endpoint_id == endpoint_id).values(status=False, deleted=True))
    db.commit()
    return {"status": 200, "message": "Endpoints and associated rsult records marked as deleted"}


@router.get("/trials/", response_model=list[TrialResponse], status_code=status.HTTP_201_CREATED)
async def trials(organisation_id: int, offset: int = 0, limit: int = Query(default=10, le=100), db: Session = Depends(get_db_session)):
    return db.exec(select(Trial).where(Trial.organisation_id == organisation_id, Trial.status == True, Trial.deleted == False).offset(offset).limit(limit)).all()


@router.get("/trial-type/", response_model=list[TrialTypeResponse], status_code=status.HTTP_201_CREATED)
async def trial_type(offset: int = 0, limit: int = Query(default=10, le=100), db: Session = Depends(get_db_session)):
    return db.exec(select(TrialType).where(TrialType.status == True, TrialType.deleted == False).offset(offset).limit(limit)).all()


@router.get("/trial-design/", response_model=list[TrialDesignResponse], status_code=status.HTTP_201_CREATED)
async def trial_type(offset: int = 0, limit: int = Query(default=10, le=100), db: Session = Depends(get_db_session)):
    return db.exec(select(TrialDesign).where(TrialDesign.status == True, TrialDesign.deleted == False).offset(offset).limit(limit)).all()


@router.get("/get-trial-data/", status_code=status.HTTP_200_OK)
async def get_trial_data(trial_file_id: int, db: Session = Depends(get_db_session)):
    query = select(TrialData.row, TrialData.attribute_key, TrialData.attribute_value).where(TrialData.trial_file_id == trial_file_id, TrialData.status == True, TrialData.deleted == False)
    result = db.exec(query).all()
    df = pd.DataFrame(result, columns=["row", "attribute_key", "attribute_value"])
    if df.empty:
        return {"message": "No data found"}
    df = df.pivot(index="row", columns="attribute_key", values="attribute_value").reset_index(drop=True)
    return df.to_dict(orient="records") 


@router.get("/get_trial_file/", status_code=status.HTTP_200_OK)
async def get_trial_file(trial_id: int, db: Session = Depends(get_db_session)):
    try:
        trial_record = db.exec(select(Trial).where(Trial.id ==trial_id, Trial.status == True, Trial.deleted == False)).first()
        if not trial_record:
            raise HTTPException(status_code=404, detail="Trial not found")

        file_records = db.exec(select(TrialFile).where(TrialFile.trial_id == trial_id, 
                                                       TrialFile.status == True, TrialFile.deleted == False)).all()
        
        column_headers = file_records[0].file_headers if file_records else []
        if not file_records:
            raise HTTPException(status_code=404, detail="No files found for the given trial ID")

        combined_dataframe = await get_trial_rec(file_records, trial_record)
        return {
                "status": 200,
                "message": "Success",
                "data": {
                    "file_id": "",      # Remove later
                    "name": trial_record.name,
                    "row_count": combined_dataframe.shape[0],
                    "rows": column_headers   # update key as "rows" into "column" later
                }
            }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {e}")


@router.get("/get-trial-data-headers/", status_code=status.HTTP_200_OK)
async def get_trial_data_headers(trial_id: int, offset: int = 0, limit: int = Query(default=10, le=100), db: Session = Depends(get_db_session)):
	file_headers = db.exec(select(TrialFile.file_headers).where(TrialFile.trial_id == trial_id, TrialFile.status == True, TrialFile.deleted == False).offset(offset).limit(limit)).first()
	return file_headers or  {"message": "No data found"}


@router.get("/get-trial-result/", status_code=status.HTTP_200_OK)
async def get_trial_result(trial_id: int, offset: int = 0, limit: int = Query(default=100, le=1000), db: Session = Depends(get_db_session)):
    file_results = db.exec(
        select(TrialResult)
        .where(TrialResult.trial_id == trial_id, TrialResult.status == True, TrialResult.deleted == False)
        .order_by(TrialResult.id.desc())
        .offset(offset)
        .limit(limit)
        ).all() 
    return {"status": 200, "message": "Success", "data": file_results}


@router.get("/endpoint-trial-result/", status_code=status.HTTP_200_OK)
async def get_trial_result(trial_id: int, endpoint_id: int, offset: int = 0, limit: int = Query(default=100, le=1000), db: Session = Depends(get_db_session)):
    file_results = db.exec(
        select(TrialResult)
        .where(TrialResult.trial_id == trial_id, TrialResult.endpoint_id == endpoint_id, TrialResult.status == True, TrialResult.deleted == False)
        .order_by(TrialResult.id.asc())
        .offset(offset)
        .limit(limit)
        ).all() 
    return {"status": 200, "message": "Success", "data": file_results}


@router.get("/get_trials/")
async def get_trials(organisation_id: int, trial_id: Optional[int] = None, db: Session = Depends(get_db_session)):
    trials = db.exec(select(Trial).options(selectinload(Trial.trial_type), selectinload(Trial.trial_design)).where(and_(or_(and_(Trial.organisation_id == organisation_id, trial_id == None),Trial.id == trial_id),Trial.status == True)).order_by(Trial.created_at.desc())).all()

    count_query = select(
        func.count().filter(Trial.status).label("active"),
        func.count().filter(not Trial.status).label("inactive"),
        func.count().filter(Trial.status, func.date_part('year', Trial.created_at) == func.date_part('year', func.now())).label("active_this_year"),
        func.count().filter(not Trial.status, func.date_part('year', Trial.created_at) == func.date_part('year', func.now())).label("inactive_this_year"),
    ).where(or_(and_(Trial.organisation_id == organisation_id, trial_id == None), Trial.id == trial_id))
    
    counts = db.exec(count_query).one()

    rows = []
    for trial in trials:
        endpoints = db.exec(select(TrialEndPoint).where(TrialEndPoint.trial_id == trial.id, TrialEndPoint.status == True, TrialEndPoint.deleted == False).order_by(TrialEndPoint.id)).all()
        formatted_endpoints = [TrialEndPointResponse.model_validate(ep, from_attributes=True).model_dump() for ep in endpoints]
        rec_to_stop_result = db.exec(select(TrialResult).where(TrialResult.trial_id == trial.id, TrialResult.rec_to_stop == True, TrialResult.status == True, TrialResult.deleted == False).order_by(TrialResult.id)).first()

        trial_data = {
            **trial.__dict__,
            "trial_type": trial.trial_type.name if trial.trial_type else None,
            "trial_design": trial.trial_design.name if trial.trial_design else None,
            "endpoints": formatted_endpoints,
            "rec_to_stop": True if rec_to_stop_result else False,
            "result_id": rec_to_stop_result.id if rec_to_stop_result else None
        }
        rows.append(TrialResponse.model_validate(trial_data, from_attributes=True).model_dump())


    return {
        "status": 200,
        "message": "Success",
        "data": {
            **counts._asdict(),
            "rows": rows
            }
        }


@router.post("/generate_plots/")
async def generate_plots(trial_req: TrialPlotRequest, db: Session = Depends(get_db_session)):
    try:
        bayes_result, result, avg_validation, avg_result, avg_data, ttest_result, km_plot_data, system_generation_data, scaled_bart_model_result, scaled_rf_model_result = {}, {}, None, None, None, None, {}, None, {}, {}

        trial_record = db.exec(select(Trial).where(Trial.id == trial_req.trial_id, Trial.status == True, Trial.deleted == False)).first()
        if not trial_record:
            raise HTTPException(status_code=404, detail="Trial not found")

        file_records = db.exec(select(TrialFile).where(TrialFile.trial_id == trial_req.trial_id, TrialFile.status == True, TrialFile.deleted == False)).all()
        if not file_records:
            raise HTTPException(status_code=404, detail="No files found for the given trial ID")
        
        selected_columns = [column.strip() for column in trial_req.columns.split(",")]

        if trial_req.cumulative_dates:
            start_date = trial_req.cumulative_dates.get("start_date")
            end_date = trial_req.cumulative_dates.get("end_date")
            week_start_date = trial_req.cumulative_dates.get("week_start_date")
            system_generation_data = {"start_date": start_date, "end_date": end_date, "week_start_date": week_start_date}
            get_week_dataframe = await get_custome_trial_rec(file_records, week_start_date, end_date)
            week_dataframe = get_week_dataframe[selected_columns]
            week_dataframe = week_dataframe.dropna()
        else:
            start_date = trial_record.start_date
            end_date = trial_record.end_date
            week_start_date = None

        combined_dataframe = await get_custome_trial_rec(file_records, start_date, end_date)
        combined_dataframe_with_all_data = combined_dataframe.copy()
        if missing_columns := [column for column in selected_columns if column not in combined_dataframe]:
            raise HTTPException(status_code=400, detail=f"Invalid columns: {', '.join(missing_columns)}")

        filtered_dataframe = combined_dataframe[selected_columns]
        selected_dataframe = filtered_dataframe.copy() # for mean_avg
        plot_spearman_heatmap_data = await plot_spearman_heatmap_with_pvalues(combined_dataframe) # correlogram 

        if trial_req.endpoint_data and not trial_req.endpoint_data.is_primary:
            primary_endpoint = db.exec(select(TrialEndPoint).where(TrialEndPoint.trial_id == trial_req.trial_id, TrialEndPoint.is_primary == True, TrialEndPoint.status == True, TrialEndPoint.deleted == False).order_by(TrialEndPoint.id.desc())).first()
            primary_grouping = next((col for col in primary_endpoint.column_data or [] if col.get("group") == "Primary Grouping"), None)
            endpoint = next((col for col in primary_endpoint.column_data or [] if col.get("group") == "Endpoint"), None)
            column_data = trial_req.endpoint_data.column_data
            selected_columns.insert(0, primary_grouping['variable_name'])
            selected_columns.append(endpoint["variable_name"])
            primary_endpoint_df = combined_dataframe[[primary_grouping['variable_name'], endpoint["variable_name"]]].dropna()
            combined_dataframe = combined_dataframe[selected_columns]
            selected_dataframe = combined_dataframe.copy()  # for mean_avg
            filtered_dataframe = await generate_custom_identifiers(combined_dataframe, selected_columns, column_data, endpoint['variable_name'])
            week_dataframe = await generate_custom_identifiers(get_week_dataframe, selected_columns, column_data, endpoint['variable_name'])
            primary_group_name = primary_grouping['variable_name']    # For KM plots   
            endpoint_group_name = endpoint['variable_name']    # For KM plots   
            endpoint_coloum_data = normalize_column_data(trial_req.endpoint_data.column_data) + normalize_column_data(primary_endpoint.column_data)   # For KM plots   

            filtering_coloum = selected_columns.copy()
            filtered_dataframe_with_date = combined_dataframe_with_all_data[filtering_coloum + ["date"]].dropna()
            
            scaled_rf_model_result = await StandardScalerAndRandomForestClassifier.run(filtered_dataframe_with_date, primary_group_name, endpoint_group_name, endpoint)
            scaled_bart_model_result = await StandardScalerAndBARTClassifier.run(filtered_dataframe_with_date, primary_group_name, endpoint_group_name, endpoint)

        if trial_req.endpoint_data.is_primary:
            primary_grouping = next((col for col in trial_req.endpoint_data.column_data or [] if col.group == "Primary Grouping"), None)
            endpoint = next((col for col in trial_req.endpoint_data.column_data or [] if col.group == "Endpoint"), None)
            primary_endpoint_df = combined_dataframe[[primary_grouping.variable_name, endpoint.variable_name]].dropna()
            primary_group_name = primary_grouping.variable_name    # For KM plots   
            endpoint_group_name = endpoint.variable_name    # For KM plots
            endpoint_coloum_data = normalize_column_data(trial_req.endpoint_data.column_data)   # For KM plots   

        if trial_req.endpoint_data:
            to_remove = next((col for col in endpoint_coloum_data if col.get("variable_name") == primary_group_name), None)
            if to_remove:
                endpoint_coloum_data.remove(to_remove)
            # km_plot_generator = GenerateKMPlots() # Generate KM plots   
            # km_plot_data = await km_plot_generator.generate_event_observed(filtered_dataframe_with_date, primary_group_name, endpoint_coloum_data)

            mean_calculation_data = await mean_calculation(selected_dataframe, endpoint_coloum_data, primary_group_name)

        if filtered_dataframe.empty:
            raise HTTPException(status_code=404, detail="No valid data found for the given date range")

        column_types = filtered_dataframe.dtypes.astype(str).tolist()
        row_count = filtered_dataframe.shape[0]
        filtered_dataframe = filtered_dataframe.dropna()
        dataframe_copy = filtered_dataframe.copy()
        
         # Case 1: One continuous and one categorical variable
        if len(filtered_dataframe.columns) == 2 and ('float' in column_types or 'float64' in column_types or 'int' in column_types or 'Int64' in column_types or 'int64' in column_types) and 'object' in column_types:
            value = await case_1_continuous_categorical(filtered_dataframe)
            if value.get("status") == "error":
                result = value['message']
            else:
                result = await case_1_graph(filtered_dataframe, value)
                bayes_result = result.get("data", {}).get("bayes_pairwise_corr_comparisons")
                if not bayes_result:
                    bayes_result = result.get("data", {}).get("bayes_factors_data", {})
                if trial_req.endpoint_data:
                    avg_data = await cal_avg(trial_req, dataframe_copy, bayes_result, week_dataframe, db)

            analyzer = TrialStatAnalyzer(primary_endpoint_df, trial_record.alpha)
            ttest_result = await analyzer.run_ttest()
            
        # Case 2: Multiple categorical variables
        elif len(filtered_dataframe.columns) >= 2 and all(ctype == 'object' for ctype in column_types):
            result = await case_2_multiple_categorical(filtered_dataframe," file_data.name")
            if result.get("status") == "error":
                result = result['message']
            else:
                bayes_result = result.get('data', {}).get('output', {}).get('bayes_factors', {})
                if trial_req.endpoint_data:
                    avg_data = await cal_avg(trial_req, dataframe_copy, bayes_result, week_dataframe, db)

            analyzer = TrialStatAnalyzer(primary_endpoint_df, trial_record.alpha)
            ttest_result = await analyzer.run_chi2()
        
        # Case 3: Two continuous variables
        elif len(filtered_dataframe.columns) == 2 and all(ctype in ['float64', 'float', 'int64', 'Int64'] for ctype in column_types):
            value =  await case_3_two_continuous(filtered_dataframe)
            if value.get("status") == "error":
                result = result['message']
            else:
                drug_data, placebo_data, drug_col_name, placebo_col_name= value["data"]["col1"], value["data"]["col2"], filtered_dataframe.columns[0],  filtered_dataframe.columns[1]
                result = await case_3_graph(drug_data, placebo_data, drug_col_name, placebo_col_name, value)
                bayes_result = result.get('data', {}).get('output', {}).get('bayesian_value_corr', {})
        
        # Case 4: Multiple categorical variables with one continuous variablecolumn_types
        elif len(filtered_dataframe.columns) >= 3 and 'object' in column_types and ('float' in column_types or 'float64' in column_types or 'int' in column_types or 'Int64' in column_types or 'int64' in column_types):
            result =  await case_4_multiple_categorical_continuous(filtered_dataframe)
            if result.get("status") == "error":
                result = result['message']
            else:
                bayes_result = result.get('data', {}).get('bayes_factor', {})
            
        # Case 5: Discontinuous data with crosstabulation
        elif len(filtered_dataframe.columns) == 2 and all(ctype == 'object' for ctype in column_types):
            result = await case_5_discontinuous_crosstab(filtered_dataframe)
            value = result

        if avg_data:
            avg_validation = avg_data['message']
            avg_result = avg_data['values']

        result["data"].update({
                            "ttest_result": ttest_result.get("pairwise_stats", {}),
                            "non_inferiority": ttest_result.get("non_inferiority_test", {}),
                            # "km_plot_data": km_plot_data if km_plot_data else {},
                            "plot_spearman_heatmap_data": plot_spearman_heatmap_data if plot_spearman_heatmap_data else {},
                            "scaled_model_result":
                                    {
                                        "bart_summary": scaled_bart_model_result if scaled_bart_model_result else {},
                                        "rf_summary": scaled_rf_model_result if scaled_rf_model_result else {}
                                    }
                        }) 
        avg_result.update({"master_avg": mean_calculation_data})

        data = {
                "trial_record": trial_record,
                "trial_req": trial_req,
                "bayesian_value": {"bayes_factor": bayes_result},
                "result": result,
                "system_generation_data": system_generation_data,
                "selected_columns": selected_columns,
                "avg_validation": avg_validation,
                "avg_result": avg_result,
                "rec_to_stop": True if avg_validation else False,
                "db": db,
            }

        result_id = await create_trial_result(data)
        if "data" in result and isinstance(result["data"], dict):
            result["data"].update({"row_count": row_count, "trial_id": trial_record.id, "trial_name": trial_record.name, "result_id": result_id})
        
        print("graph generated successfully")
        return result or {}
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Error processing file: {error}")


@router.post("/upload_trial_data/")
async def upload_trial_data(request: Request, org_id: Optional[int] = Form(None), file_data: Optional[UploadFile] = File(None), db: Session = Depends(get_db_session)):
    try:
        content_type = request.headers.get("Content-Type", "").lower()
        file_name, file_type, df = "unknown", "", None

        if "multipart/form-data" in content_type:
            form = await request.form()
            org_id = org_id or int(form.get("org_id", 0))
            if not file_data:
                raise HTTPException(status_code=400, detail="XLSX file is required")

            df = pd.read_excel(io.BytesIO(await file_data.read()))
            df = df.dropna(axis=1, how='all')
            df = df.where(pd.notna(df), None)
            for col in df.select_dtypes(include=["float64"]).columns:
                df[col] = df[col].astype(object).where(pd.notna(df[col]), None)
                # df[col] = df[col].astype("Float64")

            file_name, file_type= file_data.filename, "xlsx"

        elif "application/json" in content_type:
            body = await request.json()
            org_id = body.get("trial_data", {}).get("basic_info", {}).get("org_id", 0)
            file_name, file_type, df = "unknown.json", "json", await parse_json(json.dumps(body.get("trial_data", {}).get("data_set", [])))
        
        elif "application/xml" in content_type or "text/xml" in content_type:
            root = ET.fromstring((await request.body()).decode("utf-8"))
            org_id_element = root.find("basic_info/org_id")
            if org_id_element is None or not org_id_element.text.strip():
                raise HTTPException(status_code=400, detail="Organisation ID is required in the XML file")
            org_id, file_name, file_type, df = int(org_id_element.text), "unknown.xml", "xml", await parse_xml(root)

        if not org_id or df is None or df.empty:
            raise HTTPException(status_code=400, detail="Invalid input: Organisation ID and valid file content are required")
        
        organisation = db.exec(select(Organisation).where(Organisation.id == org_id, Organisation.status == True, Organisation.deleted == False)).first()
        if not organisation:
            raise HTTPException(status_code=404, detail="Organisation not found")

        trial_data = await create_trial_file(org_id, file_name, df, file_type, db)
        file_data = await create_trial_data(trial_data['trial_file'].id, df, db)
        
        endpoints = db.exec(select(TrialEndPoint).where(TrialEndPoint.trial_id ==trial_data['trial_id'], TrialEndPoint.status == True, TrialEndPoint.deleted == False)).all()
        if endpoints:
            db.exec(delete(TrialResult).where(TrialResult.trial_id == trial_data['trial_id'], TrialResult.is_system_generated == True))
            db.commit()
            for endpoint in endpoints:
                await generate_weekly_result(trial_data['trial_id'], endpoint, db)

        return {
            "status": 200, 
            "message": "Success", 
            "data" : {
                "file_id": trial_data['trial_file'].id,
                "trial_id": trial_data['trial_id'],
                "trial_name" : trial_data['trial_name'],
                "count" : trial_data['row_count'],
                "rows": trial_data['unique_headers']
            }
        }    
    except Exception as e:
        error_details = traceback.format_exc()
        raise HTTPException(status_code=400, detail=f"Error processing file: {error_details}")


# @router.websocket("/ws/chat/{result_id}")
# async def chat_websocket(websocket: WebSocket, result_id: int, db: Session = Depends(get_db_session), redis_client = Depends(get_redis_client)):
#     await websocket.accept()
#     await asyncio.sleep(0.1)  

#     trial_result = db.exec(select(TrialResult).where(TrialResult.id == result_id)).first()
#     trial_record = db.exec(select(Trial).where(Trial.id == trial_result.trial_id, Trial.status == True, Trial.deleted == False)).first() if trial_result else None
#     if not trial_result or not trial_record:
#         return await send_error_and_close(websocket, "TrialResult or Trial not found.")
        
#     file_records = db.exec(select(TrialFile).where(TrialFile.trial_id == trial_result.trial_id, TrialFile.status == True, TrialFile.deleted == False)).all()
#     if not file_records:
#         return await send_error_and_close(websocket, "No files found for the given trial ID")
        
#     combined_dataframe = await get_trial_rec(file_records, trial_record)
#     selected_columns = [col.strip() for col in trial_result.columns.split(",")]
#     missing = [col for col in selected_columns if col not in combined_dataframe]

#     if missing:
#         return await send_error_and_close(websocket, f"Invalid columns: {', '.join(missing)}")

#     filtered_dataframe = combined_dataframe[selected_columns].dropna()
#     if filtered_dataframe.empty:
#         return await send_error_and_close(websocket, "No valid data found for the given date range")

#     columns_data = filtered_dataframe.to_dict(orient="list")

#     await load_chat_history_from_db_to_redis(db, redis_client, result_id, websocket)

#     if redis_client.llen(f"chat_buffer:{result_id}") == 0:
#         try:
#             system_message = {
#                 "role": "system",
#                 "content": (
#                     "You are an AI assistant that explains statistical concepts in a simple, easy-to-understand way. "
#                     "Avoid code, formulas, or jargon. Use real-life analogies. Answer only statistical questions."
#                     "If a question is irrelevant (e.g., personal, political, general knowledge), politely refuse to answer."
#                     "Keep answers short and crispy—within one or two."
#                     "Only the answer should relevant to Bayes Factor."
#                 )
#             }
#             user_message = {
#                 "role": "user",
#                 "content": (
#                     f"The user has analyzed the following columns:\n{columns_data}\n"
#                     f"The computed Bayes factor is: {trial_result.bayesian_value}.\n"
#                     "Explain what this means in simple terms."
#                 )
#             }

#             response = ollama.chat(model="llama3.2:latest", messages=[system_message, user_message])
#             assistant_reply = response["message"]["content"]

#             await buffer_message_to_redis(redis_client, result_id, "user", user_message["content"])
#             await buffer_message_to_redis(redis_client, result_id, "assistant", assistant_reply)

#             send_success = await safe_send_json(websocket, {
#                 "role": "assistant",
#                 "content": assistant_reply,
#                 "timestamp": datetime.utcnow().isoformat()
#             })
#             if not send_success:
#                 return

#         except WebSocketDisconnect:
#             print(f"WebSocket disconnected during initial message for TrialResult {result_id}")
#             return
#         except Exception as e:
#             print(f"Error during initial chat generation: {e}")
#             return

#     try:
#         while True:
#             try:
#                 data = await websocket.receive_json()
#             except WebSocketDisconnect:
#                 print(f"WebSocket disconnected: TrialResult {result_id}")
#                 break
#             except Exception as e:
#                 print(f"Error while receiving message: {e}")
#                 break

#             user_input = data.get("content", "")
#             timestamp = datetime.utcnow().isoformat()
            
#             await buffer_message_to_redis(redis_client, result_id, "user", user_input)

#             history_for_reply = stream_chat_history_from_redis(redis_client, result_id)
#             response = ollama.chat(model="llama3.2:latest", messages=history_for_reply)
#             assistant_reply = response["message"]["content"]

#             await buffer_message_to_redis(redis_client, result_id, "assistant", assistant_reply)

#             send_success = await safe_send_json(websocket, {
#                 "role": "assistant",
#                 "content": assistant_reply,
#                 "timestamp": timestamp
#             })

#             if not send_success:
#                 break

#     finally:
#         await flush_messages_from_redis_to_db(redis_client, result_id, db, websocket)


@router.get("/get-chat/", response_model=list[TrialChatHistoryResponse], status_code=status.HTTP_201_CREATED)
async def trial_type(result_id: int, offset: int = 0, limit: int = Query(default=10, le=100), db: Session = Depends(get_db_session)):
    query_offset = offset + 1
    return db.exec(select(TrialResultChatHistory).where(TrialResultChatHistory.result_id == result_id, TrialResultChatHistory.status == True, TrialResultChatHistory.deleted == False).order_by(TrialResultChatHistory.id).offset(query_offset).limit(limit)).all()


@router.websocket("/ws/chat/")
async def chat_websocket(websocket: WebSocket, redis_client = Depends(get_redis_client)):
    await websocket.accept()
    await asyncio.sleep(0.1)  
    chat_session_id = generate_chat_session_id()
    try:
        while True:
            try:
                data = await websocket.receive_json()
            except WebSocketDisconnect:
                print(f"WebSocket disconnected: TrialResult")
                break
            except Exception as e:
                print(f"Error while receiving message: {e}")
                break

            user_input = data.get("content", "")
            timestamp = datetime.utcnow().isoformat()
            
            await buffer_message_to_redis(redis_client, chat_session_id, "user", user_input)

            history_for_reply = stream_chat_history_from_redis(redis_client, chat_session_id)
            response = ollama.chat(model="llama3.2:latest", messages=history_for_reply)
            assistant_reply = response["message"]["content"]

            await buffer_message_to_redis(redis_client, chat_session_id, "assistant", assistant_reply)

            send_success = await safe_send_json(websocket, {
                "role": "assistant",
                "content": assistant_reply,
                "timestamp": timestamp
            })

            if not send_success:
                break
    finally:
        await flush_messages_from_redis(redis_client, chat_session_id, websocket)
        try:
            await websocket.close(code=1000)
        except Exception as e:
            print(f"WebSocket already closed: {e}")


@router.delete("/delete-all-trials", status_code=204)
def delete_all_trials(db: Session = Depends(get_db_session)):
    db.exec(delete(TrialResultChatHistory))
    db.exec(delete(TrialResult))
    db.exec(delete(TrialData))
    db.exec(delete(TrialFile))
    db.exec(delete(TrialEndPoint))
    db.exec(delete(Trial))
    db.commit()
    return {"detail": "All trial-related records deleted successfully"}


@router.post("/get_category_unique_value/", status_code=status.HTTP_200_OK)
async def get_category_unique_value(request: TrialUniqueValueRequest, db: Session = Depends(get_db_session)):
    try:
        trial_record = db.exec(select(Trial).where(Trial.id ==request.trial_id, Trial.status == True, Trial.deleted == False)).first()
        if not trial_record:
            raise HTTPException(status_code=404, detail="Trial not found")

        file_records = db.exec(select(TrialFile).where(TrialFile.trial_id == request.trial_id, 
                                                       TrialFile.status == True, TrialFile.deleted == False)).all()
        
        column_headers = file_records[0].file_headers if file_records else []
        if not file_records:
            raise HTTPException(status_code=404, detail="No files found for the given trial ID")

        combined_dataframe = await get_trial_rec_without_date_filter(file_records)

        selected_columns = [column.strip() for column in request.columns.split(",")]
        if missing_columns := [column for column in selected_columns if column not in combined_dataframe]:
            raise HTTPException(status_code=400, detail=f"Invalid columns: {', '.join(missing_columns)}")

        filtered_dataframe = combined_dataframe[selected_columns]
        if filtered_dataframe.empty:
            raise HTTPException(status_code=404, detail="No valid data found for the given date range")

        column_types = filtered_dataframe.dtypes.astype(str).tolist()
        row_count = filtered_dataframe.shape[0]
        filtered_dataframe = filtered_dataframe.dropna()

        data = {}
        for column in filtered_dataframe.columns:
            if filtered_dataframe[column].dtype == 'object' or filtered_dataframe[column].dtype == 'string':
                unique_values = filtered_dataframe[column].dropna().unique().tolist()
                data[column] = unique_values

        return {
                "status": 200,
                "message": "Success",
                "data": data
            }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {e}")


@router.websocket("/ws/trial_alerts/{org_id}")
async def trial_alert_websocket(org_id: int, websocket: WebSocket, db: Session = Depends(get_db_session)):
    await validate_entity(db, Organisation, org_id, "Organisation not found", 404)
    await websocket.accept()
    redis = await Redis.from_url("redis://localhost")
    pubsub: PubSub = redis.pubsub()
    channel_name = f"trial_updates:{org_id}"
    try:
        await pubsub.subscribe(channel_name)
        while True:
            try:
                message = await asyncio.wait_for(
                    pubsub.get_message(ignore_subscribe_messages=True),
                    timeout=1.0
                )
                if message:
                    data = json.loads(message["data"])
                    await websocket.send_json(data)
            except asyncio.TimeoutError:
                continue
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for org {org_id}")
    except Exception as e:
        print(f"Redis listen error: {e}")
    finally:
        try:
            await pubsub.unsubscribe(channel_name)
            await redis.close()
            try:
                await websocket.close()
            except RuntimeError as e:
                print(f"WebSocket already closed: {e}")
        except Exception as cleanup_error:
            print(f"Cleanup error: {cleanup_error}")
        print("Cleaned up WebSocket & Redis connections.") 





# async def get_chatbot_data(result_id, websocket, db: Session):
#     # try:
#         trial_result = db.exec(select(TrialResult).where(TrialResult.id == result_id)).first()
#         trial_record = db.exec(select(Trial).where(Trial.id == trial_result.trial_id, Trial.status == True, Trial.deleted == False)).first() if trial_result else None
#         if not trial_result or not trial_record:
#             return await send_error_and_close(websocket, "TrialResult or Trial not found.")
            
#         file_records = db.exec(select(TrialFile).where(TrialFile.trial_id == trial_result.trial_id, TrialFile.status == True, TrialFile.deleted == False)).all()
#         if not file_records:
#             return await send_error_and_close(websocket, "No files found for the given trial ID")
            
#         combined_dataframe = await get_trial_rec(file_records, trial_record)
#         selected_columns = [col.strip() for col in trial_result.columns.split(",")]
#         missing = [col for col in selected_columns if col not in combined_dataframe]

#         if missing:
#             return await send_error_and_close(websocket, f"Invalid columns: {', '.join(missing)}")

#         filtered_dataframe = combined_dataframe[selected_columns].dropna()
#         if filtered_dataframe.empty:
#             return await send_error_and_close(websocket, "No valid data found for the given date range")
        
#         columns_data = filtered_dataframe.to_dict(orient="list")

#         return columns_data, trial_result
#     # except Exception as e:
#     #         print(f"Error during initial chat generation: {e}")
#     #         return


# @router.websocket("/ws/chat/{result_id}")
# async def chat_websocket(websocket: WebSocket, result_id: int, db: Session = Depends(get_db_session), redis_client = Depends(get_redis_client)):
#     await websocket.accept()
#     await asyncio.sleep(0.1)  

    

#     columns_data, trial_result = await get_chatbot_data(result_id, websocket, db)

#     print("columns_data", columns_data)

#     await load_chat_history_from_db_to_redis(db, redis_client, result_id, websocket)

#     if redis_client.llen(f"chat_buffer:{result_id}") == 0:
#         try:
#             system_message = {
#                 "role": "system",
#                 "content": (
#                     "You are an AI assistant that explains statistical concepts in a simple, easy-to-understand way. "
#                     "Avoid code, formulas, or jargon. Use real-life analogies. Answer only statistical questions."
#                     "If a question is irrelevant (e.g., personal, political, general knowledge), politely refuse to answer."
#                     "Keep answers short and crispy—within one or two."
#                     "Only the answer should relevant to Bayes Factor."
#                 )
#             }
#             user_message = {
#                 "role": "user",
#                 "content": (
#                     f"The user has analyzed the following columns:\n{columns_data}\n"
#                     f"The computed Bayes factor is: {trial_result.bayesian_value}.\n"
#                     "Explain what this means in simple terms."
#                 )
#             }

#             response = ollama.chat(model="llama3.2:latest", messages=[system_message, user_message])
#             assistant_reply = response["message"]["content"]

#             await buffer_message_to_redis(redis_client, result_id, "user", user_message["content"])
#             await buffer_message_to_redis(redis_client, result_id, "assistant", assistant_reply)

#             send_success = await safe_send_json(websocket, {
#                 "role": "assistant",
#                 "content": assistant_reply,
#                 "timestamp": datetime.utcnow().isoformat()
#             })
#             if not send_success:
#                 return

#         except WebSocketDisconnect:
#             print(f"WebSocket disconnected during initial message for TrialResult {result_id}")
#             return
#         except Exception as e:
#             print(f"Error during initial chat generation: {e}")
#             return

#     try:
#         while True:
#             try:
#                 data = await websocket.receive_json()
#             except WebSocketDisconnect:
#                 print(f"WebSocket disconnected: TrialResult {result_id}")
#                 break
#             except Exception as e:
#                 print(f"Error while receiving message: {e}")
#                 break

#             user_input = data.get("content", "")
#             timestamp = datetime.utcnow().isoformat()
            
#             await buffer_message_to_redis(redis_client, result_id, "user", user_input)

#             history_for_reply = stream_chat_history_from_redis(redis_client, result_id)
#             response = ollama.chat(model="llama3.2:latest", messages=history_for_reply)
#             assistant_reply = response["message"]["content"]

#             await buffer_message_to_redis(redis_client, result_id, "assistant", assistant_reply)

#             send_success = await safe_send_json(websocket, {
#                 "role": "assistant",
#                 "content": assistant_reply,
#                 "timestamp": timestamp
#             })

#             if not send_success:
#                 break

#     finally:
#         await flush_messages_from_redis_to_db(redis_client, result_id, db, websocket)



# @router.websocket("/ws/chat/{result_id}")
# async def chat_websocket(websocket: WebSocket, result_id: int, db: Session = Depends(get_db_session), redis_client = Depends(get_redis_client)):
#     import io
#     import contextlib
#     from langchain.llms import Ollama
#     from langchain.agents import initialize_agent, Tool
#     from langchain.agents.agent_types import AgentType
#     from langchain.agents.tools import tool

#     await websocket.accept()
#     await asyncio.sleep(0.1)  

#     trial_result = db.exec(select(TrialResult).where(TrialResult.id == result_id)).first()
#     trial_record = db.exec(select(Trial).where(Trial.id == trial_result.trial_id, Trial.status == True, Trial.deleted == False)).first() if trial_result else None
#     if not trial_result or not trial_record:
#         return await send_error_and_close(websocket, "TrialResult or Trial not found.")
        
#     file_records = db.exec(select(TrialFile).where(TrialFile.trial_id == trial_result.trial_id, TrialFile.status == True, TrialFile.deleted == False)).all()
#     if not file_records:
#         return await send_error_and_close(websocket, "No files found for the given trial ID")
        
#     combined_dataframe = await get_trial_rec(file_records, trial_record)
#     print("combined_dataframe", combined_dataframe)
#     print("trial_result.columns.split(","), ", trial_result.columns.split(","))
#     # selected_columns = [col.strip() for col in trial_result.columns.split(",")]
#     selected_columns = trial_result.columns.strip("{} ").split(",")
#     selected_columns = [col.strip() for col in selected_columns]
#     missing = [col for col in selected_columns if col not in combined_dataframe]
#     print("missing", missing)

#     if missing:
#         return await send_error_and_close(websocket, f"Invalid columns: {', '.join(missing)}")

#     filtered_dataframe = combined_dataframe[selected_columns].dropna()
#     if filtered_dataframe.empty:
#         return await send_error_and_close(websocket, "No valid data found for the given date range")

#     columns_data = filtered_dataframe.to_dict(orient="list")


#     @tool
#     def query_dataframe(code: str) -> str:
#         """Executes a query on a pandas DataFrame named `df`."""
#         try:
#             with contextlib.redirect_stdout(io.StringIO()) as f:
#                 result = eval(code)
#             output = f.getvalue()
#             return output.strip() + "\n" + str(result)
#         except Exception as e:
#             return f"Error: {str(e)}"

#     # Step 3: Initialize Ollama LLM + agent
#     llm = Ollama(model="llama3")
#     tools = [Tool(name="query_dataframe", func=query_dataframe, description="Run Python code to query a DataFrame called `df`.")]

#     agent = initialize_agent(
#         tools=tools,
#         llm=llm,
#         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#         verbose=True
#     )

#     # Step 4: Enter WebSocket receive/send loop
#     try:
#         while True:
#             data = await websocket.receive_text()
#             if data.lower() in {"exit", "quit"}:
#                 break

#             user_query = f"Given a pandas DataFrame named `df`, answer this: {data}. Write and run Python code to get the answer."

#             result = agent.run(user_query)
#             await websocket.send_text(result)

#     except Exception as e:
#         await websocket.send_text(f"Error during chat: {str(e)}")
#     finally:
#         await websocket.close()



















#     await load_chat_history_from_db_to_redis(db, redis_client, result_id, websocket)

#     if redis_client.llen(f"chat_buffer:{result_id}") == 0:
#         try:
#             system_message = {
#                 "role": "system",
#                 "content": (
#                     "You are an AI assistant that explains statistical concepts in a simple, easy-to-understand way. "
#                     "Avoid code, formulas, or jargon. Use real-life analogies. Answer only statistical questions."
#                     "If a question is irrelevant (e.g., personal, political, general knowledge), politely refuse to answer."
#                     "Keep answers short and crispy—within one or two."
#                     "Only the answer should relevant to Bayes Factor."
#                 )
#             }
#             user_message = {
#                 "role": "user",
#                 "content": (
#                     f"The user has analyzed the following columns:\n{columns_data}\n"
#                     f"The computed Bayes factor is: {trial_result.bayesian_value}.\n"
#                     "Explain what this means in simple terms."
#                 )
#             }

#             response = ollama.chat(model="llama3.2:latest", messages=[system_message, user_message])
#             assistant_reply = response["message"]["content"]

#             await buffer_message_to_redis(redis_client, result_id, "user", user_message["content"])
#             await buffer_message_to_redis(redis_client, result_id, "assistant", assistant_reply)

#             send_success = await safe_send_json(websocket, {
#                 "role": "assistant",
#                 "content": assistant_reply,
#                 "timestamp": datetime.utcnow().isoformat()
#             })
#             if not send_success:
#                 return

#         except WebSocketDisconnect:
#             print(f"WebSocket disconnected during initial message for TrialResult {result_id}")
#             return
#         except Exception as e:
#             print(f"Error during initial chat generation: {e}")
#             return

#     try:
#         while True:
#             try:
#                 data = await websocket.receive_json()
#             except WebSocketDisconnect:
#                 print(f"WebSocket disconnected: TrialResult {result_id}")
#                 break
#             except Exception as e:
#                 print(f"Error while receiving message: {e}")
#                 break

#             user_input = data.get("content", "")
#             timestamp = datetime.utcnow().isoformat()
            
#             await buffer_message_to_redis(redis_client, result_id, "user", user_input)

#             history_for_reply = stream_chat_history_from_redis(redis_client, result_id)
#             response = ollama.chat(model="llama3.2:latest", messages=history_for_reply)
#             assistant_reply = response["message"]["content"]

#             await buffer_message_to_redis(redis_client, result_id, "assistant", assistant_reply)

#             send_success = await safe_send_json(websocket, {
#                 "role": "assistant",
#                 "content": assistant_reply,
#                 "timestamp": timestamp
#             })

#             if not send_success:
#                 break

#     finally:
#         await flush_messages_from_redis_to_db(redis_client, result_id, db, websocket)

