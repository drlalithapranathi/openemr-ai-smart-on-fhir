import mysql.connector

# DB connection
conn = mysql.connector.connect(
    host="127.0.0.1",
    user="openemr",       # or "root"
    password="openemr",   # or "root"
    database="openemr"
)
cursor = conn.cursor()

# Tables you want to explore
tables_to_check = [
    "patient_data","history_data","form_encounter","lists",
    "billing","external_procedures","prescriptions","drug_sales",
    "form_vitals","form_observation","form_soap","form_clinical_notes",
    "form_dictation","documents","users","procedure_order",
    "procedure_report","procedure_result","form_care_plan",
    "form_clinical_instructions","form_reviewofs","form_ros",
    "immunizations","facility"
]

with open("table_attributes_samples.txt", "w") as f:
    for table in tables_to_check:
        f.write(f"\n==== {table.upper()} ====\n")

        # Get column names
        cursor.execute(f"SHOW COLUMNS FROM {table}")
        columns = [col[0] for col in cursor.fetchall()]
        f.write("Columns: " + ", ".join(columns) + "\n\n")

        # Get 5 sample rows
        cursor.execute(f"SELECT * FROM {table} LIMIT 5")
        rows = cursor.fetchall()
        if rows:
            for row in rows:
                f.write(str(row) + "\n")
        else:
            f.write("No data found.\n")

cursor.close()
conn.close()
