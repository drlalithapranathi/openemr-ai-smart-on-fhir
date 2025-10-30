import mysql.connector

# ------------------- Database Connection -------------------
conn = mysql.connector.connect(
    host="127.0.0.1",
    user="openemr",
    password="openemr",
    database="openemr"
)
cursor = conn.cursor(dictionary=True)

pid = 5555

# ------------------- Step 1: Get the most recent encounter -------------------
cursor.execute(f"""
    SELECT encounter, date, date_end
    FROM form_encounter
    WHERE pid={pid}
    ORDER BY date DESC
    LIMIT 1
""")
encounter = cursor.fetchone()

if not encounter:
    print(f"No encounters found for patient {pid}.")
    cursor.close()
    conn.close()
    exit()

enc_id = encounter["encounter"]
start_date = encounter.get("date")
end_date = encounter.get("date_end")

print(f"Extracting data for Patient {pid} → Encounter {enc_id} ({start_date} → {end_date})")

# ------------------- Step 2: Define Queries -------------------
queries = {
    "Demographics": f"""
        SELECT fname, lname, DOB, sex, pubpid AS unit_no,
               street, city, state, postal_code
        FROM patient_data
        WHERE pid = {pid}
    """,

    "Service_Attending": f"""
        SELECT fe.encounter,
               fe.date AS admission_date,
               fe.date_end AS discharge_date,
               fe.encounter_type_description AS service,
               u.fname AS attending_fname,
               u.lname AS attending_lname,
               u.specialty
        FROM form_encounter fe
        LEFT JOIN users u ON fe.provider_id = u.id
        WHERE fe.pid = {pid} AND fe.encounter = {enc_id}
    """,

    "HPI": f"""
        SELECT subjective
        FROM form_soap
        WHERE pid = {pid} AND DATE(date) = (SELECT DATE(date) FROM form_encounter WHERE encounter={enc_id})
    """,

    "Physical_Exam": f"""
        SELECT objective
        FROM form_soap
        WHERE pid = {pid} AND DATE(date) = (SELECT DATE(date) FROM form_encounter WHERE encounter={enc_id})
    """,

    "Hospital_Course": f"""
        SELECT assessment, plan
        FROM form_soap
        WHERE pid = {pid} AND DATE(date) = (SELECT DATE(date) FROM form_encounter WHERE encounter={enc_id})
    """,

    "Past_Medical_History": f"""
        SELECT title, diagnosis, begdate, enddate, outcome
        FROM lists
        WHERE pid = {pid} AND type='medical_problem'
    """,

    "Allergies": f"""
        SELECT diagnosis AS allergy, begdate AS noted_on
        FROM lists
        WHERE pid = {pid} AND type='allergy'
    """,

    "Vitals": f"""
        SELECT date, weight, height, bps, bpd, pulse, respiration, temperature, oxygen_saturation
        FROM form_vitals
        WHERE pid = {pid} AND DATE(date) = (SELECT DATE(date) FROM form_encounter WHERE encounter={enc_id})
    """,

    "Lab_Results": f"""
        SELECT pr.result_text, pr.result, pr.units, pr.range, pr.abnormal, pr.comments, pr.result_code
        FROM procedure_order po
        JOIN procedure_report prt ON po.procedure_order_id = prt.procedure_order_id
        JOIN procedure_result pr ON prt.procedure_report_id = pr.procedure_report_id
        WHERE po.patient_id = {pid} AND po.encounter_id = {enc_id}
        ORDER BY pr.date DESC
    """,

    "Imaging_Procedures": f"""
        SELECT poc.procedure_code, poc.procedure_name, prt.report_notes
        FROM procedure_order po
        JOIN procedure_order_code poc ON po.procedure_order_id = poc.procedure_order_id
        JOIN procedure_report prt ON po.procedure_order_id = prt.procedure_order_id
        WHERE po.patient_id = {pid} AND po.encounter_id = {enc_id}
    """,

    "Medications": f"""
        SELECT drug, dosage, route, `interval`, start_date, end_date, active
        FROM prescriptions
        WHERE patient_id = {pid} AND encounter = {enc_id}
    """,

    "Care_Plan": f"""
        SELECT date, codetext, description, note_related_to
        FROM form_care_plan
        WHERE pid = {pid} AND encounter = {enc_id}
    """,

    "Discharge_Instructions": f"""
        SELECT instruction, date
        FROM form_clinical_instructions
        WHERE pid = {pid} AND encounter = {enc_id}
    """,

    "Family_History": f"""
        SELECT history_mother, history_father, history_siblings, history_offspring,
               relatives_cancer, relatives_diabetes, relatives_high_blood_pressure,
               relatives_heart_problems, relatives_stroke
        FROM history_data
        WHERE pid = {pid}
    """,

    "Social_History": f"""
        SELECT alcohol, tobacco, recreational_drugs, coffee,
               exercise_patterns, sleep_patterns, seatbelt_use
        FROM history_data
        WHERE pid = {pid}
    """
}

# ------------------- Step 3: Formatter Function -------------------
def format_row(section, row):
    if section == "Demographics":
        return (f"Patient: {row.get('fname', '')} {row.get('lname', '')} "
                f"({row.get('sex', '')}, DOB: {row.get('DOB', '')})\n"
                f"Address: {row.get('street', '')}, {row.get('city', '')}, "
                f"{row.get('state', '')} {row.get('postal_code', '')}\n"
                f"Unit #: {row.get('unit_no', '')}")

    if section == "Service_Attending":
        return (f"Encounter: {row.get('encounter', '')}\n"
                f"Admission: {row.get('admission_date', '')} → Discharge: {row.get('discharge_date', '')}\n"
                f"Service: {row.get('service', '')}\n"
                f"Attending: Dr. {row.get('attending_fname', '')} {row.get('attending_lname', '')} "
                f"({row.get('specialty', '')})")

    if section == "HPI":
        return f"HPI: {row.get('subjective', '')}"

    if section == "Physical_Exam":
        return f"Physical Exam: {row.get('objective', '')}"

    if section == "Hospital_Course":
        return (f"Assessment: {row.get('assessment', '')}\n"
                f"Plan: {row.get('plan', '')}")

    if section == "Past_Medical_History":
        return (f"{row.get('title', '')}: {row.get('diagnosis', '')} "
                f"(From {row.get('begdate', '')} → {row.get('enddate', '')}) "
                f"Outcome: {row.get('outcome', '')}")

    if section == "Allergies":
        return f"Allergy: {row.get('allergy', '')} (Noted On: {row.get('noted_on', '')})"

    if section == "Vitals":
        return (f"Vitals on {row.get('date', '')}: "
                f"BP {row.get('bps', '')}/{row.get('bpd', '')}, "
                f"Pulse {row.get('pulse', '')}, Temp {row.get('temperature', '')}, "
                f"O2 Sat {row.get('oxygen_saturation', '')}")

    if section == "Lab_Results":
        return (f"{row.get('result_text', '')}: {row.get('result', '')} {row.get('units', '')} "
                f"(Range: {row.get('range', '')}) "
                f"{'[ABNORMAL]' if row.get('abnormal') else ''}")

    if section == "Imaging_Procedures":
        return (f"Procedure: {row.get('procedure_name', '')} ({row.get('procedure_code', '')})\n"
                f"Findings: {row.get('report_notes', '')}")

    if section == "Medications":
        return (f"{row.get('drug', '')} {row.get('dosage', '')} {row.get('route', '')} every {row.get('interval', '')} "
                f"(From {row.get('start_date', '')} to {row.get('end_date', '')}) "
                f"{'Active' if row.get('active') else 'Inactive'}")

    if section == "Care_Plan":
        return (f"{row.get('codetext', '')}: {row.get('description', '')} "
                f"(Note: {row.get('note_related_to', '')})")

    if section == "Discharge_Instructions":
        return f"{row.get('instruction', '')} ({row.get('date', '')})"

    if section == "Family_History":
        return (f"Mother: {row.get('history_mother', '')}; Father: {row.get('history_father', '')}; "
                f"Siblings: {row.get('history_siblings', '')}; Offspring: {row.get('history_offspring', '')}")

    if section == "Social_History":
        return (f"Alcohol: {row.get('alcohol', '')}; Tobacco: {row.get('tobacco', '')}; "
                f"Drugs: {row.get('recreational_drugs', '')}; Coffee: {row.get('coffee', '')}; "
                f"Exercise: {row.get('exercise_patterns', '')}; Sleep: {row.get('sleep_patterns', '')}; "
                f"Seatbelt: {row.get('seatbelt_use', '')}")

    return str(row)

# ------------------- Step 4: Run Extraction -------------------
with open("patient_5555_summary.txt", "w") as f:
    f.write(f"==== Clinical Summary for Patient {pid} (Encounter {enc_id}) ====\n\n")

    for section, query in queries.items():
        f.write(f"\n-- {section} --\n")
        try:
            cursor.execute(query)
            rows = cursor.fetchall()
            if rows:
                for row in rows:
                    f.write(format_row(section, row) + "\n")
            else:
                f.write("No data found.\n")
        except mysql.connector.Error as e:
            f.write(f"Error in {section}: {e}\n")

cursor.close()
conn.close()
print("✅ Extraction complete → patient_5555_summary.txt")
