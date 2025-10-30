import mysql.connector
from tabulate import tabulate  # pretty printing (pip install tabulate)

# Database connection
conn = mysql.connector.connect(
    host="127.0.0.1",
    user="openemr",
    password="openemr",
    database="openemr"
)
cursor = conn.cursor()

# Patient ID to test with
pid = 11031

# Tables to inspect
tables = [
    # "form_encounter"
    # "form_soap"
    # "form_vitals"
    # "lists"
    "history_data"
    # "procedure_order",
    # "procedure_report",
    # "procedure_result",
    # "prescriptions"
    # "form_care_plan",
    # "form_clinical_instructions",
    # "history_data",
    # "patient_data"

]
# Demographics           - patient_data [fname, lname, DOB, sex, pubpid AS unit_no, street, city, state, postal_code]

# Family_History         - history_data [history_mother, history_father, history_siblings, history_offspring, relatives_cancer, relatives_diabetes, relatives_high_blood_pressure, relatives_heart_problems, relatives_stroke]
# Social_History         - history_data [alcohol, tobacco, recreational_drugs, coffee, exercise_patterns, sleep_patterns, seatbelt_use]

# Allergies              - lists [title AS allergy, begdate AS noted_on]
# Past_Medical_History   - lists [title, diagnosis, begdate, enddate, outcome]

# Vitals                 - form_vitals [date, weight, height, bps, bpd, pulse, respiration, temperature, oxygen_saturation, BMI]
# Encounter_Info         - form_encounter, users [fe.encounter,fe.date AS admission_date, fe.date_end AS discharge_date, fe.reason AS chief_complaint, fe.discharge_disposition, fe.encounter_type_description AS service, u.fname AS attending_fname, u.lname AS attending_lname, u.specialty]
# HPI                    - form_encounter, form_soap [fe.encounter, fe.date, fs.subjective]

# Physical_Exams         - form_encounter, form_soap [fe.encounter, fe.date, fs.objective]
# Hospital_Course        - form_encounter, form_soap [fe.encounter, fe.date, fs.assessment, fs.plan]

# Lab_Results            - procedure_order, procedure_report, procedure_result [po.encounter_id AS encounter, pr.result_text, pr.result, pr.units, pr.range, pr.abnormal, pr.comments, pr.result_code]
# Imaging_Procedures     - procedure_order, procedure_report, procedure_order_code [po.encounter_id AS encounter, poc.procedure_code, poc.procedure_name, prt.report_notes, prt.date_report]

# Medications            - prescriptions  [encounter, drug,rxnorm_drugcode, dosage, route, `interval`, quantity, refills, start_date, end_date, indication, provider_id, active, note]
# Care_Plan 	         - form_care_plan [encounter, date, codetext, description, note_related_to]
# Discharge_Instructions - form_clinical_instructions [encounter, instruction, date]

for table in tables:
    print(f"\n========== {table.upper()} ==========\n")

    # 1. Schema
    cursor.execute(f"DESCRIBE {table}")
    schema = cursor.fetchall()
    print("Schema:")
    print(tabulate(schema, headers=["Field", "Type", "Null", "Key", "Default", "Extra"], tablefmt="psql"))

    # 2. Sample rows
    try:
        cursor.execute(f"SELECT * FROM {table} WHERE pid = {pid} LIMIT 5")
    except mysql.connector.Error:
        # Not all tables have pid (but in OpenEMR most do)
        cursor.execute(f"SELECT * FROM {table} LIMIT 5")
    rows = cursor.fetchall()

    print("\nSample rows:")
    if rows:
        columns = [desc[0] for desc in cursor.description]
        print(tabulate(rows, headers=columns, tablefmt="psql"))
    else:
        print("No rows found.")

cursor.close()
conn.close()
