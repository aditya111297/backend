import io
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from app.model_loader import registry
from app.routers.dashboard import get_dashboard_metrics
from app.routers.models import get_feature_importance, get_models_performance
from app.routers.risk import get_risk_analysis

router = APIRouter(prefix="/api/reports", tags=["Reports"])


@router.get("/export")
def export_report(format: str = Query(default="csv", pattern="^(csv|pdf)$")):
    if registry.active_dataset_df is None:
        raise HTTPException(status_code=404, detail="No active dataset available for report generation.")

    dash = get_dashboard_metrics()
    try:
        models_data = get_models_performance()["models"]
    except HTTPException:
        models_data = []

    try:
        fi_data = get_feature_importance(top_n=10)["top_10"]
    except HTTPException:
        fi_data = []

    try:
        risk_data = get_risk_analysis(limit=25)
    except HTTPException:
        risk_data = {"summary": {}, "customers": []}

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if format == "csv":
        out = io.StringIO()
        out.write("# ChurnAnalytics AI - Executive Churn & Model Report\n")
        out.write(f"# Generated: {timestamp}\n")
        out.write(f"# Dataset: {dash['dataset_name']}\n\n")

        out.write("## 1. DATASET & CHURN SUMMARY\n")
        summary_df = pd.DataFrame([
            {"Metric": "Total Customers", "Value": dash["total_customers"]},
            {"Metric": "Churned Customers", "Value": dash["churned_customers"]},
            {"Metric": "Active Customers", "Value": dash["active_customers"]},
            {"Metric": "Churn Percentage (%)", "Value": dash["churn_percentage"]},
            {"Metric": "Average Tenure", "Value": dash["average_customer_tenure"]},
            {"Metric": "High-Risk Customers", "Value": dash["high_risk_customers"]},
            {"Metric": "Best Performing Model", "Value": dash["best_performing_model"]},
        ])
        summary_df.to_csv(out, index=False)
        out.write("\n")

        if models_data:
            out.write("## 2. MODEL COMPARISON RESULTS\n")
            pd.DataFrame(models_data).to_csv(out, index=False)
            out.write("\n")

        if fi_data:
            out.write("## 3. TOP 10 PERMUTATION FEATURE IMPORTANCE\n")
            pd.DataFrame(fi_data).to_csv(out, index=False)
            out.write("\n")

        if risk_data.get("customers"):
            out.write("## 4. TOP HIGH-RISK CUSTOMERS\n")
            risk_rows = [
                {
                    "Customer_ID": c["customer_id"],
                    "Churn_Probability": c["churn_probability"],
                    "Risk_Category": c["risk_category"],
                    "Recommended_Action": " | ".join(c["recommendations"][:2]),
                }
                for c in risk_data["customers"]
            ]
            pd.DataFrame(risk_rows).to_csv(out, index=False)

        out.seek(0)
        return StreamingResponse(
            iter([out.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="ChurnAnalytics_Report.csv"'},
        )

    # Generate PDF Report via ReportLab
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("ChurnAnalytics AI – Executive Predictive Analytics Report", styles["Title"]))
    elements.append(Paragraph(f"Generated: {timestamp} | Dataset: {dash['dataset_name']}", styles["Normal"]))
    elements.append(Spacer(1, 14))

    # Section 1: Summary Table
    elements.append(Paragraph("1. Dataset & Churn Summary", styles["Heading2"]))
    s_table_data = [
        ["Metric", "Value"],
        ["Total Customers", str(dash["total_customers"])],
        ["Churned Customers", str(dash["churned_customers"])],
        ["Active Customers", str(dash["active_customers"])],
        ["Churn Rate", f"{dash['churn_percentage']}%"],
        ["Average Customer Tenure", str(dash["average_customer_tenure"] or "N/A")],
        ["High-Risk Customers", str(dash["high_risk_customers"])],
        ["Best Model", str(dash["best_performing_model"] or "N/A")],
    ]
    t1 = Table(s_table_data, colWidths=[240, 260])
    t1.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4338CA")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
    ]))
    elements.append(t1)
    elements.append(Spacer(1, 14))

    # Section 2: Model Comparison
    if models_data:
        elements.append(Paragraph("2. Machine Learning Model Comparison", styles["Heading2"]))
        headers = list(models_data[0].keys())
        m_rows = [headers] + [[str(round(v, 4)) if isinstance(v, float) else str(v) for v in row.values()] for row in models_data]
        t2 = Table(m_rows)
        t2.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4338CA")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        elements.append(t2)
        elements.append(Spacer(1, 14))

    # Section 3: Feature Importance
    if fi_data:
        elements.append(Paragraph("3. Top 10 Permutation Feature Importance", styles["Heading2"]))
        fi_rows = [["Rank", "Feature", "Importance Score"]] + [
            [str(r["rank"]), r["feature"], str(r["importance"])] for r in fi_data
        ]
        t3 = Table(fi_rows, colWidths=[60, 280, 160])
        t3.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4338CA")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        elements.append(t3)
        elements.append(Spacer(1, 14))

    # Section 4: Top High-Risk Customers
    if risk_data.get("customers"):
        elements.append(Paragraph("4. Priority Risk Segment (Top 15 Customers)", styles["Heading2"]))
        r_rows = [["Customer ID", "Churn Prob.", "Risk Category"]] + [
            [c["customer_id"], f"{round(c['churn_probability']*100, 1)}%", c["risk_category"]]
            for c in risk_data["customers"][:15]
        ]
        t4 = Table(r_rows, colWidths=[180, 140, 180])
        t4.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4338CA")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        elements.append(t4)

    doc.build(elements)
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="ChurnAnalytics_Report.pdf"'},
    )
