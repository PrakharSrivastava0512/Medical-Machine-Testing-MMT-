import os
import io
import sqlite3
import base64
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, session, send_file, flash
import pandas as pd
import numpy as np
import joblib

# Use non-interactive Matplotlib backend to prevent GUI threads issues
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

app = Flask(__name__)
app.secret_key = "medipredict_secure_encryption_key_2026"

# Ensure outputs directory exists
os.makedirs('outputs', exist_ok=True)
DATABASE = 'outputs/history.db'

# Load ML models and encoders on startup
try:
    failure_classifier = joblib.load('models/failure_classifier.pkl')
    rul_regressor = joblib.load('models/rul_regressor.pkl')
    label_encoders = joblib.load('models/label_encoders.pkl')
    print("All machine learning models and encoders loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    print("Please make sure you have run 'train_model.py' to generate model picks.")

# Database initialization
def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                equipment_id TEXT NOT NULL,
                equipment_type TEXT NOT NULL,
                manufacturer TEXT NOT NULL,
                age_months REAL NOT NULL,
                usage_hours_per_day REAL NOT NULL,
                total_scans_or_uses INTEGER NOT NULL,
                days_since_last_maintenance REAL NOT NULL,
                num_previous_repairs INTEGER NOT NULL,
                avg_temperature_c REAL NOT NULL,
                max_temperature_c REAL NOT NULL,
                vibration_level_mm_s REAL NOT NULL,
                voltage_fluctuation_pct REAL NOT NULL,
                coolant_level_pct REAL NOT NULL,
                helium_level_pct REAL NOT NULL,
                tube_current_ma REAL NOT NULL,
                error_logs_last_30_days INTEGER NOT NULL,
                power_supply_stability_score REAL NOT NULL,
                ambient_humidity_pct REAL NOT NULL,
                component_wear_index REAL NOT NULL,
                technician_rating_last_inspection REAL NOT NULL,
                failure_probability REAL NOT NULL,
                remaining_useful_life REAL NOT NULL,
                health_score REAL NOT NULL,
                risk_level TEXT NOT NULL,
                recommendation TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()

init_db()

# Admin credentials (static for demo purposes)
ADMIN_USER = "admin"
ADMIN_PASS = "hospital123"

# Context processor to inject active navigation tag
@app.context_processor
def inject_active():
    return dict(active_page=request.endpoint)

# Helper: Recommendation generator based on sensors & risk
def generate_recommendation(risk, eq_type, mfg, wear, maintenance_days, vibration, coolant, helium, errors, voltage):
    recs = []
    
    # Specific sensor alarms
    if vibration > 3.0:
        recs.append("Misalignment Alert: Vibration level exceeds normal bounds. Calibrate motor mounts.")
    if coolant < 85.0 and eq_type in ['CT Scanner', 'MRI Machine']:
        recs.append("Thermal Risk: Coolant level depleted. Top off reservoir immediately.")
    if helium < 90.0 and eq_type == 'MRI Machine':
        recs.append("Cryogenic Danger: Helium level is low. Check compression vacuum seal.")
    if errors > 4:
        recs.append("Interface Issue: Excessive error logs detected. Run firmware diagnostics.")
    if voltage > 5.0:
        recs.append("Power Drift: Fluctuations detected. Verify power stabilizer functions.")
    
    # Base recommendation based on overall Risk Level
    if risk == 'Critical':
        recs.append("CRITICAL CONDITION: Decommission the equipment immediately. Dispatch emergency engineering crew for parts replacement.")
    elif risk == 'High':
        recs.append("HIGH RISK: Schedule service within 48 hours. Avoid running high-stress diagnostic protocols.")
    elif risk == 'Moderate':
        recs.append("MODERATE RISK: Plan regular maintenance check within the next 14 days. Monitor telemetry trends closely.")
    else:
        recs.append("LOW RISK: System operates within normal parameters. Continue standard routine inspection schedule.")
        
    return " | ".join(recs)

# Helper: Health score calculation logic
def calculate_health_score(wear, maintenance_days):
    # Base health: 100 - component wear index
    health = 100.0 - wear
    
    # Maintenance bonus/penalty
    if maintenance_days <= 30:
        bonus = 8.0 # Bonus for recently serviced machines
    elif maintenance_days <= 90:
        bonus = 4.0
    elif maintenance_days > 180:
        bonus = -8.0 # Penalty for overdue maintenance
    elif maintenance_days > 365:
        bonus = -15.0 # High penalty
    else:
        bonus = 0.0
        
    # Bound health score between 0 and 100
    return max(0.0, min(100.0, health + bonus))

# Helper: Matplotlib plot to base64 string
def fig_to_base64(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight', dpi=100)
    img.seek(0)
    base64_str = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close(fig)
    return base64_str

# ==========================================
# ROUTES
# ==========================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('prediction.html', result=None, form_values=None)
    
    # Post processing
    try:
        # Retrieve form data
        form_data = {
            'equipment_id': request.form.get('equipment_id', '').strip(),
            'equipment_type': request.form.get('equipment_type'),
            'manufacturer': request.form.get('manufacturer'),
            'age_months': float(request.form.get('age_months', 0)),
            'usage_hours_per_day': float(request.form.get('usage_hours_per_day', 0)),
            'total_scans_or_uses': int(request.form.get('total_scans_or_uses', 0)),
            'days_since_last_maintenance': float(request.form.get('days_since_last_maintenance', 0)),
            'num_previous_repairs': int(request.form.get('num_previous_repairs', 0)),
            'avg_temperature_c': float(request.form.get('avg_temperature_c', 0)),
            'max_temperature_c': float(request.form.get('max_temperature_c', 0)),
            'vibration_level_mm_s': float(request.form.get('vibration_level_mm_s', 0)),
            'voltage_fluctuation_pct': float(request.form.get('voltage_fluctuation_pct', 0)),
            'coolant_level_pct': float(request.form.get('coolant_level_pct', 100.0)),
            'helium_level_pct': float(request.form.get('helium_level_pct', 100.0)),
            'tube_current_ma': float(request.form.get('tube_current_ma', 0.0)),
            'error_logs_last_30_days': int(request.form.get('error_logs_last_30_days', 0)),
            'power_supply_stability_score': float(request.form.get('power_supply_stability_score', 100.0)),
            'ambient_humidity_pct': float(request.form.get('ambient_humidity_pct', 0)),
            'component_wear_index': float(request.form.get('component_wear_index', 0)),
            'technician_rating_last_inspection': float(request.form.get('technician_rating_last_inspection', 0))
        }

        # Encode categorical features
        enc_eq_type = label_encoders['equipment_type'].transform([form_data['equipment_type']])[0]
        enc_mfg = label_encoders['manufacturer'].transform([form_data['manufacturer']])[0]

        # Prepare feature vector matching training schema
        feature_vector = np.array([[
            enc_eq_type,
            enc_mfg,
            form_data['age_months'],
            form_data['usage_hours_per_day'],
            form_data['total_scans_or_uses'],
            form_data['days_since_last_maintenance'],
            form_data['num_previous_repairs'],
            form_data['avg_temperature_c'],
            form_data['max_temperature_c'],
            form_data['vibration_level_mm_s'],
            form_data['voltage_fluctuation_pct'],
            form_data['coolant_level_pct'],
            form_data['helium_level_pct'],
            form_data['tube_current_ma'],
            form_data['error_logs_last_30_days'],
            form_data['power_supply_stability_score'],
            form_data['ambient_humidity_pct'],
            form_data['component_wear_index'],
            form_data['technician_rating_last_inspection']
        ]])

        # Model predictions
        failure_prob = float(failure_classifier.predict_proba(feature_vector)[0][1])
        rul = float(rul_regressor.predict(feature_vector)[0])
        
        # Adjust RUL logically: it cannot be negative
        rul = max(0.0, rul)

        # Health score logic
        health_score = calculate_health_score(form_data['component_wear_index'], form_data['days_since_last_maintenance'])

        # Risk Classification Logic
        # Rules: Failure >= 60% OR RUL <= 20 days -> Critical
        # Failure >= 35% -> High
        # Failure >= 15% -> Moderate
        # Else -> Low
        if failure_prob >= 0.60 or rul <= 20.0:
            risk_level = 'Critical'
            risk_class = 'risk-critical'
        elif failure_prob >= 0.35:
            risk_level = 'High'
            risk_class = 'risk-high'
        elif failure_prob >= 0.15:
            risk_level = 'Moderate'
            risk_class = 'risk-moderate'
        else:
            risk_level = 'Low'
            risk_class = 'risk-low'

        recommendation = generate_recommendation(
            risk_level, 
            form_data['equipment_type'], 
            form_data['manufacturer'],
            form_data['component_wear_index'],
            form_data['days_since_last_maintenance'],
            form_data['vibration_level_mm_s'],
            form_data['coolant_level_pct'],
            form_data['helium_level_pct'],
            form_data['error_logs_last_30_days'],
            form_data['voltage_fluctuation_pct']
        )

        # Save to database history log
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO prediction_history (
                    equipment_id, equipment_type, manufacturer, age_months, usage_hours_per_day,
                    total_scans_or_uses, days_since_last_maintenance, num_previous_repairs,
                    avg_temperature_c, max_temperature_c, vibration_level_mm_s,
                    voltage_fluctuation_pct, coolant_level_pct, helium_level_pct,
                    tube_current_ma, error_logs_last_30_days, power_supply_stability_score,
                    ambient_humidity_pct, component_wear_index, technician_rating_last_inspection,
                    failure_probability, remaining_useful_life, health_score, risk_level, recommendation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                form_data['equipment_id'], form_data['equipment_type'], form_data['manufacturer'],
                form_data['age_months'], form_data['usage_hours_per_day'], form_data['total_scans_or_uses'],
                form_data['days_since_last_maintenance'], form_data['num_previous_repairs'],
                form_data['avg_temperature_c'], form_data['max_temperature_c'], form_data['vibration_level_mm_s'],
                form_data['voltage_fluctuation_pct'], form_data['coolant_level_pct'], form_data['helium_level_pct'],
                form_data['tube_current_ma'], form_data['error_logs_last_30_days'], form_data['power_supply_stability_score'],
                form_data['ambient_humidity_pct'], form_data['component_wear_index'], form_data['technician_rating_last_inspection'],
                failure_prob, rul, health_score, risk_level, recommendation
            ))
            conn.commit()
            last_id = cursor.lastrowid

        result = {
            'id': last_id,
            'equipment_id': form_data['equipment_id'],
            'equipment_type': form_data['equipment_type'],
            'manufacturer': form_data['manufacturer'],
            'failure_probability': failure_prob,
            'remaining_useful_life': rul,
            'health_score': health_score,
            'risk_level': risk_level,
            'risk_class': risk_class,
            'recommendation': recommendation,
            'date': datetime.now().strftime('%Y-%m-%d %H:%M')
        }

        return render_template('prediction.html', result=result, form_values=form_data)
        
    except Exception as ex:
        flash(f"Diagnostics prediction failed: {str(ex)}", "danger")
        return render_template('prediction.html', result=None, form_values=request.form)

@app.route('/dashboard')
def dashboard():
    # Load dataset
    try:
        base_df = pd.read_csv('medical_equipment_data.csv')
    except Exception:
        base_df = pd.DataFrame()
        
    # Read history database log entries
    with get_db() as conn:
        history_df = pd.read_sql_query('SELECT * FROM prediction_history', conn)

    # 1. Total statistics calculations
    base_total = len(base_df)
    hist_total = len(history_df)
    total_tested = base_total + hist_total
    
    # Calculate averages from bases and merges
    if not history_df.empty:
        # Weighted metric merges
        avg_health = history_df['health_score'].mean()
        avg_rul = history_df['remaining_useful_life'].mean()
        # Count critical assets: Failure probability >= 60% OR RUL <= 20
        critical_count = len(history_df[(history_df['failure_probability'] >= 0.6) | (history_df['remaining_useful_life'] <= 20)])
    else:
        # Defaults if database is empty (derive from base dataset)
        # Component wear index is in base dataset. Est base health score:
        base_health = (100.0 - base_df['component_wear_index']).mean()
        avg_health = base_health
        avg_rul = base_df['remaining_useful_life_days'].mean()
        critical_count = len(base_df[(base_df['failure_within_30_days'] == 1) | (base_df['remaining_useful_life_days'] <= 20)])

    stats = {
        'total_tested': total_tested,
        'critical_count': critical_count,
        'avg_health': avg_health,
        'avg_rul': avg_rul
    }

    # Combined dataset for plotting distributions
    plot_df = base_df.copy()
    if not history_df.empty:
        # Rename columns to match base df
        hist_rename = history_df.rename(columns={
            'failure_probability': 'failure_within_30_days',
            'remaining_useful_life': 'remaining_useful_life_days'
        })
        # Concatenate
        plot_df = pd.concat([plot_df, hist_rename], ignore_index=True)

    # Use a professional color palette
    plt.rcParams['text.color'] = '#1e293b'
    plt.rcParams['axes.labelcolor'] = '#1e293b'
    plt.rcParams['xtick.color'] = '#1e293b'
    plt.rcParams['ytick.color'] = '#1e293b'
    
    # Ensure plots have transparent or custom backgrounds
    plot_theme_color = ['#2a6f97', '#014f86', '#00a896', '#f59e0b', '#f97316', '#ef4444']

    # --- Plot 1: Equipment Distribution ---
    fig1, ax1 = plt.subplots(figsize=(6, 4.5))
    eq_counts = plot_df['equipment_type'].value_counts()
    ax1.pie(eq_counts, labels=eq_counts.index, autopct='%1.1f%%', colors=plot_theme_color, 
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5, 'antialiased': True})
    ax1.axis('equal')
    eq_dist_b64 = fig_to_base64(fig1)

    # --- Plot 2: Failure Probability Distribution ---
    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    # For baseline, failure target is 0/1. For logs, it is floats (probabilities).
    # Handle float probability checks safely
    ax2.hist(plot_df['failure_within_30_days'], bins=15, color='#2a6f97', edgecolor='white', alpha=0.85)
    ax2.set_xlabel('Failure Probability')
    ax2.set_ylabel('Asset Count')
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    failure_dist_b64 = fig_to_base64(fig2)

    # --- Plot 3: Feature Importance (From Hybrid models) ---
    fig3, ax3 = plt.subplots(figsize=(6, 4.5))
    feature_names = [
        'equipment_type', 'manufacturer', 'age_months', 'usage_hours_per_day', 
        'total_scans_or_uses', 'days_since_last_maintenance', 'num_previous_repairs', 
        'avg_temperature_c', 'max_temperature_c', 'vibration_level_mm_s', 
        'voltage_fluctuation_pct', 'coolant_level_pct', 'helium_level_pct', 
        'tube_current_ma', 'error_logs_last_30_days', 'power_supply_stability_score', 
        'ambient_humidity_pct', 'component_wear_index', 'technician_rating_last_inspection'
    ]
    # Get importance from Random Forest (estimator 0) in the loaded Voting Classifier
    try:
        rf_clf = failure_classifier.named_estimators_['rf']
        importances = rf_clf.feature_importances_
        indices = np.argsort(importances)[-10:] # Top 10
        ax3.barh(range(10), importances[indices], color='#00a896', align='center')
        ax3.set_yticks(range(10))
        ax3.set_yticklabels([feature_names[i] for i in indices])
        ax3.set_xlabel('Relative Variable Importance')
    except Exception:
        # Fallback dummy bar chart in case extraction fails
        ax3.barh(range(3), [0.4, 0.3, 0.2], color='#00a896', align='center')
        ax3.set_yticks(range(3))
        ax3.set_yticklabels(['wear_index', 'maintenance_days', 'vibration'])
        
    feature_importance_b64 = fig_to_base64(fig3)

    # --- Plot 4: RUL Distribution ---
    fig4, ax4 = plt.subplots(figsize=(6, 4.5))
    ax4.hist(plot_df['remaining_useful_life_days'].dropna(), bins=15, color='#468faf', edgecolor='white', alpha=0.85)
    ax4.set_xlabel('Remaining Useful Life (Days)')
    ax4.set_ylabel('Asset Count')
    ax4.grid(axis='y', linestyle='--', alpha=0.3)
    rul_dist_b64 = fig_to_base64(fig4)

    # --- Plot 5: Risk Level Distribution ---
    fig5, ax5 = plt.subplots(figsize=(6, 4.5))
    # Categorize base dataset to merge risk distributions
    # Apply standard risk rule to compile risk profile
    temp_risks = []
    for idx, row in plot_df.iterrows():
        prob = row['failure_within_30_days']
        rul_val = row['remaining_useful_life_days']
        if prob >= 0.60 or rul_val <= 20:
            temp_risks.append('Critical')
        elif prob >= 0.35:
            temp_risks.append('High')
        elif prob >= 0.15:
            temp_risks.append('Moderate')
        else:
            temp_risks.append('Low')
            
    risk_counts = pd.Series(temp_risks).value_counts()
    # Order them logically
    risk_order = ['Low', 'Moderate', 'High', 'Critical']
    risk_counts = risk_counts.reindex(risk_order).fillna(0)
    
    ax5.bar(risk_counts.index, risk_counts.values, color=['#10b981', '#f59e0b', '#f97316', '#ef4444'], edgecolor='none', alpha=0.85)
    ax5.set_ylabel('Asset Count')
    ax5.set_xlabel('Risk Class Category')
    ax5.grid(axis='y', linestyle='--', alpha=0.3)
    risk_dist_b64 = fig_to_base64(fig5)

    # --- Plot 6: Monthly Wear & Failure Trend ---
    fig6, ax6 = plt.subplots(figsize=(6, 4.5))
    # Group by age intervals to simulate a cumulative degradation path
    # Group age in months into bins of 6 months
    bins = np.arange(0, plot_df['age_months'].max() + 6, 6)
    plot_df['age_bin'] = pd.cut(plot_df['age_months'], bins=bins, labels=bins[:-1])
    trend = plot_df.groupby('age_bin', observed=False)['component_wear_index'].mean()
    ax6.plot(trend.index.astype(float), trend.values, marker='o', linewidth=2.5, color='#ef4444', label='Component Wear')
    ax6.set_xlabel('Equipment Operating Age (Months)')
    ax6.set_ylabel('Avg Wear Index (%)')
    ax6.legend(loc='upper left')
    ax6.grid(linestyle='--', alpha=0.3)
    trend_dist_b64 = fig_to_base64(fig6)

    charts = {
        'equipment_dist': eq_dist_b64,
        'failure_dist': failure_dist_b64,
        'feature_importance': feature_importance_b64,
        'rul_dist': rul_dist_b64,
        'risk_dist': risk_dist_b64,
        'trend_dist': trend_dist_b64
    }

    return render_template('dashboard.html', stats=stats, charts=charts)

@app.route('/history')
def history():
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM prediction_history ORDER BY timestamp DESC')
        logs = [dict(row) for row in cursor.fetchall()]
    return render_template('history.html', logs=logs)

# Admin Authentication Handlers
@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')
    
    if username == ADMIN_USER and password == ADMIN_PASS:
        session['logged_in'] = True
        flash("Authentication successful. Administrator session active.", "success")
    else:
        flash("Invalid administrator credentials. Access denied.", "danger")
        
    return redirect(request.referrer or url_for('history'))

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    flash("Administrator session closed.", "info")
    return redirect(url_for('history'))

@app.route('/delete_log/<int:log_id>', methods=['POST'])
def delete_log(log_id):
    if not session.get('logged_in'):
        flash("Unauthorized action. Please log in first.", "danger")
        return redirect(url_for('history'))
        
    with get_db() as conn:
        conn.execute('DELETE FROM prediction_history WHERE id = ?', (log_id,))
        conn.commit()
    flash("Diagnostic log entry deleted successfully.", "success")
    return redirect(url_for('history'))

@app.route('/model_info')
def model_info():
    return render_template('model_info.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/export_pdf/<int:prediction_id>')
def export_pdf(prediction_id):
    # Fetch diagnostic logs
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM prediction_history WHERE id = ?', (prediction_id,))
        log = cursor.fetchone()
        
    if not log:
        flash("PDF Report generation failed: Record not found.", "danger")
        return redirect(url_for('history'))
    
    # Generate ReportLab PDF in memory
    buffer = io.BytesIO()
    
    # Establish document layouts
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=letter,
        rightMargin=36, 
        leftMargin=36,
        topMargin=36, 
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    # Define custom professional styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        textColor=colors.HexColor('#014f86'),
        spaceAfter=12
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#64748b'),
        spaceAfter=20
    )
    
    section_title = ParagraphStyle(
        'SectionTitle',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        textColor=colors.HexColor('#2a6f97'),
        spaceBefore=12,
        spaceAfter=6
    )
    
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#1e293b')
    )
    
    metric_title_style = ParagraphStyle(
        'MetricTitle',
        parent=styles['Normal'],
        fontSize=8,
        fontName='Helvetica-Bold',
        textColor=colors.HexColor('#64748b'),
        alignment=1 # Centered
    )
    
    metric_val_style = ParagraphStyle(
        'MetricVal',
        parent=styles['Normal'],
        fontSize=12,
        fontName='Helvetica-Bold',
        textColor=colors.HexColor('#1e293b'),
        alignment=1 # Centered
    )

    story = []
    
    # PDF Header Bar
    story.append(Paragraph("MediPredict AI — Diagnostic & Prognosis Report", title_style))
    story.append(Paragraph(f"Official Compliance Diagnostic Brief | Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", subtitle_style))
    story.append(Spacer(1, 10))
    
    # 1. Device Info Table
    story.append(Paragraph("1. Device Specifications", section_title))
    device_data = [
        [Paragraph("<b>Parameter</b>", body_style), Paragraph("<b>Value</b>", body_style), Paragraph("<b>Parameter</b>", body_style), Paragraph("<b>Value</b>", body_style)],
        [Paragraph("Equipment ID", body_style), Paragraph(log['equipment_id'], body_style), Paragraph("Equipment Type", body_style), Paragraph(log['equipment_type'], body_style)],
        [Paragraph("Manufacturer", body_style), Paragraph(log['manufacturer'], body_style), Paragraph("Age (Months)", body_style), Paragraph(str(log['age_months']), body_style)],
        [Paragraph("Daily Usage (Hours)", body_style), Paragraph(str(log['usage_hours_per_day']), body_style), Paragraph("Total Uses/Scans", body_style), Paragraph(str(log['total_scans_or_uses']), body_style)]
    ]
    t1 = Table(device_data, colWidths=[130, 130, 130, 130])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
    ]))
    story.append(t1)
    story.append(Spacer(1, 12))
    
    # 2. Sensor Telemetry Table
    story.append(Paragraph("2. Diagnostic Telemetry & Sensors", section_title))
    sensor_data = [
        [Paragraph("<b>Sensor Parameter</b>", body_style), Paragraph("<b>Reading</b>", body_style), Paragraph("<b>Sensor Parameter</b>", body_style), Paragraph("<b>Reading</b>", body_style)],
        [Paragraph("Avg Temp (°C)", body_style), Paragraph(f"{log['avg_temperature_c']} °C", body_style), Paragraph("Max Temp (°C)", body_style), Paragraph(f"{log['max_temperature_c']} °C", body_style)],
        [Paragraph("Vibration Level (mm/s)", body_style), Paragraph(f"{log['vibration_level_mm_s']} mm/s", body_style), Paragraph("Voltage Fluctuation", body_style), Paragraph(f"{log['voltage_fluctuation_pct']}%", body_style)],
        [Paragraph("Coolant Level (%)", body_style), Paragraph(f"{log['coolant_level_pct']}%", body_style), Paragraph("Helium Level (%)", body_style), Paragraph(f"{log['helium_level_pct']}%", body_style)],
        [Paragraph("X-Ray Tube Current (mA)", body_style), Paragraph(f"{log['tube_current_ma']} mA", body_style), Paragraph("Power Supply Stability", body_style), Paragraph(f"{log['power_supply_stability_score']}%", body_style)],
        [Paragraph("Component Wear Index", body_style), Paragraph(f"{log['component_wear_index']}%", body_style), Paragraph("Technician Rating", body_style), Paragraph(f"{log['technician_rating_last_inspection']}%", body_style)],
    ]
    t2 = Table(sensor_data, colWidths=[130, 130, 130, 130])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
    ]))
    story.append(t2)
    story.append(Spacer(1, 12))
    
    # 3. Model Predictions Cards represented as Table
    story.append(Paragraph("3. AI Diagnostics & Prognosis Results", section_title))
    
    # Set colors based on risk
    risk_color_hex = '#10b981' # default low
    if log['risk_level'] == 'Critical':
        risk_color_hex = '#ef4444'
    elif log['risk_level'] == 'High':
        risk_color_hex = '#f97316'
    elif log['risk_level'] == 'Moderate':
        risk_color_hex = '#f59e0b'
        
    metric_data = [
        [
            Paragraph("FAILURE PROBABILITY (30d)", metric_title_style),
            Paragraph("MACHINE HEALTH SCORE", metric_title_style),
            Paragraph("REMAINING USEFUL LIFE (RUL)", metric_title_style),
            Paragraph("RISK LEVEL", metric_title_style)
        ],
        [
            Paragraph(f"<font color='{risk_color_hex}'><b>{log['failure_probability'] * 100:.2f}%</b></font>", metric_val_style),
            Paragraph(f"<font color='#014f86'><b>{log['health_score']:.1f}/100</b></font>", metric_val_style),
            Paragraph(f"<b>{log['remaining_useful_life']:.1f} Days</b>", metric_val_style),
            Paragraph(f"<font color='{risk_color_hex}'><b>{log['risk_level'].upper()}</b></font>", metric_val_style)
        ]
    ]
    t3 = Table(metric_data, colWidths=[130, 130, 130, 130])
    t3.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f8fafc')),
        ('GRID', (0,0), (-1,-1), 1, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(t3)
    story.append(Spacer(1, 15))
    
    # 4. Clinical Recommendation block
    story.append(Paragraph("4. Maintenance & Clinical Action Recommendations", section_title))
    rec_text = log['recommendation']
    
    rec_box_style = ParagraphStyle(
        'RecBox',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#0f172a'),
    )
    
    t4 = Table([[Paragraph(f"<b>STATUS REPORT:</b> {rec_text}", rec_box_style)]], colWidths=[520])
    t4.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor(risk_color_hex)),
        ('LINEBEFORE', (0,0), (0,-1), 4, colors.HexColor(risk_color_hex)),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('LEFTPADDING', (0,0), (-1,-1), 15),
        ('RIGHTPADDING', (0,0), (-1,-1), 15),
    ]))
    story.append(t4)
    story.append(Spacer(1, 30))
    
    # 5. Signatures Block
    sig_data = [
        [
            Paragraph("<b>Diagnostic Technician Signature:</b>", body_style),
            Paragraph("<b>Medical Administrator Signature:</b>", body_style)
        ],
        [
            Spacer(1, 20),
            Spacer(1, 20)
        ],
        [
            Paragraph("_____________________________<br/>Biomedical Engineering Dept.", body_style),
            Paragraph("_____________________________<br/>Operations & Facilities Director", body_style)
        ]
    ]
    t5 = Table(sig_data, colWidths=[260, 260])
    t5.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'BOTTOM'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t5)
    
    # Build Document
    doc.build(story)
    buffer.seek(0)
    
    pdf_filename = f"MediPredict_Report_{log['equipment_id']}_{datetime.now().strftime('%Y%m%d')}.pdf"
    
    return send_file(
        buffer, 
        as_attachment=True, 
        download_name=pdf_filename, 
        mimetype='application/pdf'
    )

if __name__ == '__main__':
    # Initialize DB table and load server
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)
