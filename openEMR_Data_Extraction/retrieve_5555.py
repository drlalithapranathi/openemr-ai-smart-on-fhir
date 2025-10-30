import mysql.connector
conn = mysql.connector.connect(
    host="127.0.0.1",
    user="openemr",
    password="openemr",
    database="openemr"
)
cursor = conn.cursor(dictionary=True)
pid = 11042

# Taylor Kozey     (Osteoarthritis) - 11038
# Heath Maggio     (Gall Stones)    - 11039
# Rakesh Sharma    (COPD)           - 11040
# Bhavana Sharma   (Anemia)         - 11041
# Toma Kurtz       (Diabetes)       - 11042
# Nicholas Schimdt (GERD)           - 11043


# Things not considered
# Family history: Spouse,Epilepsy, Suicide, Tuberculosis , Mental Illness
# Social history: Counselling, Hazardous Activities, Seatbelt Use

def format_row(section: str, row: dict) -> str:
    """
    Turn a dict row into a human-readable string.
    Handles some sections with special formatting.
    """
    if not row:
        return "No data."

    if section == "Demographics":
        return (f"Name: {row.get('fname', '')} {row.get('lname', '')}\n"
                f"DOB: {row.get('DOB', '')}\n"
                f"Sex: {row.get('sex', '')}\n"
                f"Unit Number: {row.get('unit_no', '')}\n"
                f"Address: {row.get('street', '')}, {row.get('city', '')}, "
                f"{row.get('state', '')} {row.get('postal_code', '')}")


    if section == "Social_History":
        return (f"Coffee: {row.get('coffee', '')}\n"
                f"Tobacco: {row.get('tobacco', '')}\n"
                f"Alcohol: {row.get('alcohol', '')}\n"
                f"Recreational Drugs: {row.get('recreational_drugs', '')}\n"
                f"Exercise: {row.get('exercise_patterns', '')}\n"
                f"Sleep: {row.get('sleep_patterns', '')}\n")

    if section == "Family_History":
        return (f"Mother: {row.get('history_mother', '')}\n"
                f"Father: {row.get('history_father', '')}\n"
                f"Siblings: {row.get('history_siblings', '')}\n"
                f"Offspring: {row.get('history_offspring', '')}\n"
                f"Cancer: {row.get('relatives_cancer', '')}\n"
                f"Diabetes: {row.get('relatives_diabetes', '')}\n"
                f"Heart Problems: {row.get('relatives_heart_problems', '')}\n"
                f"Stroke: {row.get('relatives_stroke', '')}")


    if section == "Encounter_Info":
        return (
            f"Encounter: {row.get('encounter', '')}\n"
            f"Admission: {row.get('admission_date', '')} → Discharge: {row.get('discharge_date', '')}\n"
            # f"Service: {row.get('service', '')}\n"
            f"Facility: {row.get('facility', '')}\n"
            f"Chief Complaint: {row.get('chief_complaint', '')}\n"
            f"Discharge Disposition: {row.get('discharge_disposition', '')}\n"
            f"Attending: Dr. {row.get('attending_fname', '')} {row.get('attending_lname', '')} "
            f"({row.get('specialty', '')})"
        )


    # if section == "Vitals":
    #     return (f"Date: {row.get('date', '')}\n"
    #             f"Temperature: {row.get('temperature', '')}\n"
    #             f"Blood Pressure: {row.get('bps', '')}/{row.get('bpd', '')}\n"
    #             f"Heart Rate: {row.get('pulse', '')}\n"
    #             f"Respiration Rate: {row.get('respiration', '')}\n"
    #             f"SpO₂: {row.get('oxygen_saturation', '')}\n"
    #             f"Weight in kg: {row.get('weight', '')}\n"
    #             f"Height in cm: {row.get('height', '')}\n"
    #             f"BMI: {row.get('BMI', '')}")

    if section == "Vitals":
    # Helper to format numbers safely with two decimals
        def fmt(value):
            try:
                num = float(value)
                return f"{num:.2f}"
            except (TypeError, ValueError):
                return "N/A"

        return (
            f"Date: {row.get('date', 'N/A')}\n"
            f"Temperature: {fmt(row.get('temperature'))} °C\n"
            f"Blood Pressure: {fmt(row.get('bps'))}/{fmt(row.get('bpd'))} mmHg\n"
            f"Heart Rate: {fmt(row.get('pulse'))} bpm\n"
            f"Respiration Rate: {fmt(row.get('respiration'))} breaths/min\n"
            f"SpO₂: {fmt(row.get('oxygen_saturation'))} %\n"
        f"Weight: {fmt(row.get('weight'))} kg\n"
        f"Height: {fmt(row.get('height'))} cm\n"
        f"BMI: {fmt(row.get('BMI'))}\n"
    )


    if section == "Medications":
    # Safely extract all fields with defaults
        drug_name = row.get('drug', 'Unknown medication')
        rxnorm = row.get('rxnorm_drugcode', '')
        dosage = row.get('dosage', '')
        route = row.get('route', '')
        interval = row.get('interval', '')
        quantity = row.get('quantity', '')
        refills = row.get('refills', '')
        indication = row.get('indication', '')
        start_date = row.get('start_date', '')
        end_date = row.get('end_date', '')
        note = row.get('note', '')
        active_flag = row.get('active', 0)
        provider_fname = row.get('provider_fname', '')
        provider_lname = row.get('provider_lname', '')

    # Derived readable fields
        active_status = "Active" if active_flag == 1 else "Inactive"
        interval_text = f" every {interval} hrs" if interval else ""
        indication_text = f" — Indication: {indication}" if indication else ""
        rxnorm_text = f" (RxNorm: {rxnorm})" if rxnorm else ""
        prescriber = (
            f"Prescribed by Dr. {provider_fname} {provider_lname}".strip()
            if provider_fname or provider_lname
            else ""
    )

        return (
            f"{drug_name}{rxnorm_text}\n"
            f"Dosage: {dosage or 'N/A'} | Route: {route or 'N/A'}{interval_text}\n"
            f"Start: {start_date or 'N/A'} → End: {end_date or 'Ongoing'} ({active_status})\n"
            f"Quantity: {quantity or 'N/A'} | Refills: {refills or 'N/A'}{indication_text}\n"
            f"Notes: {note or 'None'}\n"
            f"{prescriber}\n"
    )


    # if section == "Medications":
    #     return (f"{row.get('drug', '')} "
    #             f"{row.get('dosage', '') or ''} "
    #             f"Route: {row.get('route', '') or ''}, "
    #             f"Interval: {row.get('interval', '') or ''}\n"
    #             f"Start: {row.get('start_date', '')} → End: {row.get('end_date', '')} "
    #             f"{'(Active)' if row.get('active') == 1 else '(Inactive)'}")

    # Glucose [Mass/volume] in Blood: 76.47 mg/dL (Range: ) {Why there is no comment on it!}
    # Cause of Death [US Standard Certificate of Death]: {entry.value} UNK (Range: ) {why death certificate is here!}
    if section == "Lab_Results":
        return (f"{row.get('result_text', '')} (LOINC: {row.get('result_code', '')}) is {row.get('result', '')}  {row.get('units', '')} "
                f"(Range: {row.get('range', '')}) "
                f"{' [ABNORMAL]' if row.get('abnormal') else ''}"
                f"(Comments: {row.get('comments', '')})")

    if section == "Imaging_Procedures":
        return (f"Procedure: {row.get('procedure_name', '')} "
                f"({row.get('procedure_code', '')})\n"
                f"Status: {row.get('order_status', '')}\n"
                f"Report Date: {row.get('date_report', '')}\n"
                f"Findings: {row.get('report_notes', '')}")


    if section == "Allergies":
        return (
            f"Allergy: {row.get('allergy', 'Unknown')} ({row.get('diagnosis', 'N/A')})\n"
            f"Reaction: {row.get('reaction', 'N/A')} \n"
            f"Severity: {row.get('severity_al', 'N/A')}\n"
            # f"Occurrence: {row.get('occurrence', 'N/A')}\n"
            f"Noted On: {row.get('noted_on', 'N/A')}\n"
            f"Comments: {row.get('comments', 'None')}\n"
        )


    if section == "HPI":
        return (f"Date: {row.get('date', '')}\n"
                f"HPI: {row.get('subjective', '')}")

    # if section == "Past_Medical_History":
    #     title = row.get('title', '')
    #     diagnosis = row.get('diagnosis', '')
    #     begdate = row.get('begdate', '')
    #     enddate = row.get('enddate', '')
    #     outcome = row.get('outcome', '')
    #
    #     # Format neatly with defaults
    #     return (
    #         f"Condition: {title or 'N/A' }\n"
    #         f"Diagnosis Code: {diagnosis or 'N/A'}\n"
    #         f"Onset Date: {begdate or 'N/A'} → End Date: {enddate or 'Ongoing'}\n"
    #         f"Outcome: {outcome or 'N/A'}\n"
    #     )

    if section == "Physical_Exam":
        return (f"Date: {row.get('date', '')}\n"
                f"Physical Exam: {row.get('objective', '')}")

    if section == "Hospital_Course":
        return (f"Date: {row.get('date', '')}\n"
                f"Assessment: {row.get('assessment', '')}\n"
                f"Plan: {row.get('plan', '')}")


    if section == "Care_Plan":
        return (f"Date: {row.get('date', '')} | Encounter: {row.get('encounter', '')}\n"
                f"Code: {row.get('codetext', '')}\n"
                f"Plan: {row.get('description', '')}\n"
                f"Notes: {row.get('note_related_to', '')}")

    if section == "Discharge_Instructions":
        return (f"Date: {row.get('date', '')}\n"
                f"Encounter: {row.get('encounter', '')}\n"
                f"Instruction: {row.get('instruction', '')}")


    # Default → print key: value pairs
    return "; ".join(f"{k}: {v}" for k, v in row.items() if v not in (None, "", 0))


# ---------------- Patient-level queries ----------------
queries_patient_level = {
    "Demographics": """
                    SELECT fname, lname, DOB, sex, pubpid AS unit_no,
                           street, city, state, postal_code
                    FROM patient_data
                    WHERE pid = {pid}
                    """,


    "Family_History": """
                      SELECT history_mother, history_father, history_siblings, history_offspring,
                             relatives_cancer, relatives_diabetes, relatives_high_blood_pressure,
                             relatives_heart_problems, relatives_stroke
                      FROM history_data
                      WHERE pid = {pid}
                      """,

    "Social_History": """
                      SELECT alcohol, tobacco, recreational_drugs, coffee,
                             exercise_patterns, sleep_patterns, seatbelt_use
                      FROM history_data
                      WHERE pid = {pid}
                      """,

    "Allergies": f"""
                    SELECT 
                        title AS allergy,
                        diagnosis,
                        occurrence,
                        severity_al,
                        begdate AS noted_on,
                        comments
                    FROM lists
                    WHERE pid = {pid} AND type = 'allergy'
                    ORDER BY begdate DESC
""",
    "Vitals" : """
    SELECT date, weight, height, bps, bpd, pulse, respiration, 
           temperature, oxygen_saturation, BMI
    FROM form_vitals
    WHERE pid = {pid}
    ORDER BY date DESC
    """
}

queries_per_encounter = {

    "Encounter_Info": f"""
    SELECT fe.encounter,
           fe.date AS admission_date,
           fe.date_end AS discharge_date,
           fe.reason AS chief_complaint,
           fe.discharge_disposition,
--            fe.encounter_type_description AS service,         # Here I can see the cheif complaint being dublicated
           u.fname AS attending_fname,
           u.lname AS attending_lname,
           u.specialty
    FROM form_encounter fe
    LEFT JOIN users u ON fe.provider_id = u.id
    WHERE fe.pid = {{pid}} AND fe.encounter = {{enc_id}}
"""
    ,
#     # VITALS
#     "Vitals": f"""
#     SELECT
#         fv.id AS vitals_id,
#         fe.encounter,
#         fv.date,
#         fv.weight,
#         fv.height,
#         fv.bps,
#         fv.bpd,
#         fv.pulse,
#         fv.respiration,
#         fv.temperature,
#         fv.oxygen_saturation,
#         fv.BMI
#     FROM form_encounter fe
#     LEFT JOIN form_vitals fv
#         ON fv.pid = fe.pid
#     WHERE fe.pid = {{pid}}
#     ORDER BY fv.date DESC
# """,


    # HPI (subjective part of SOAP)
    "HPI": f"""
        SELECT fe.encounter, fe.date, fs.subjective
        FROM form_encounter fe
        LEFT JOIN form_soap fs 
            ON fs.pid = fe.pid 
           AND DATE(fs.date) = DATE(fe.date)
        WHERE fe.pid = {{pid}}
        ORDER BY fe.date DESC
    """,

    # Past Medical History
    "Past_Medical_History": f"""
        SELECT title, diagnosis, begdate, enddate
        FROM lists
        WHERE pid = {{pid}} AND type = 'medical_problem'
    """,



    # Physical Exam (objective part of SOAP)
    "Physical_Exam": f"""
        SELECT fe.encounter, fe.date, fs.objective
        FROM form_encounter fe
        LEFT JOIN form_soap fs 
            ON fs.pid = fe.pid 
           AND DATE(fs.date) = DATE(fe.date)
        WHERE fe.pid = {{pid}}
        ORDER BY fe.date DESC
    """,

    # # Glucose [Mass/volume] in Blood: 76.47 mg/dL (Range: ) {Why there is no comment on it!}
    "Lab_Results": f"""
        SELECT po.encounter_id AS encounter,
               pr.result_text, pr.result, pr.units, pr.range, pr.abnormal, pr.comments, pr.result_code
        FROM procedure_order po
        JOIN procedure_report prt ON po.procedure_order_id = prt.procedure_order_id
        JOIN procedure_result pr ON prt.procedure_report_id = pr.procedure_report_id
        WHERE po.patient_id = {{pid}}
        ORDER BY pr.date DESC
    """,

    # "Imaging_Procedures": f"""
    #     SELECT po.encounter_id AS encounter,
    #            poc.procedure_code, poc.procedure_name, prt.report_notes
    #     FROM procedure_order po
    #     JOIN procedure_order_code poc ON po.procedure_order_id = poc.procedure_order_id
    #     JOIN procedure_report prt ON po.procedure_order_id = prt.procedure_order_id
    #     WHERE po.patient_id = {{pid}}
    #     ORDER BY prt.date_report DESC
    # """
    "Imaging_Procedures": f"""
    SELECT po.encounter_id AS encounter,
           poc.procedure_code,
           poc.procedure_name,
           prt.report_notes,
           prt.date_report
    FROM procedure_order po
    JOIN procedure_order_code poc 
        ON po.procedure_order_id = poc.procedure_order_id
    JOIN procedure_report prt 
        ON po.procedure_order_id = prt.procedure_order_id
    WHERE po.patient_id = {{pid}}
      AND (poc.procedure_type LIKE 'imaging'
           OR poc.procedure_name LIKE '%X-ray%'
           OR poc.procedure_name LIKE '%CT%'
           OR poc.procedure_name LIKE '%MRI%'
           OR poc.procedure_name LIKE '%Ultrasound%')
    ORDER BY prt.date_report DESC
""",

    "Medications": f"""
    SELECT 
        encounter,
        drug,
        rxnorm_drugcode,
        dosage,
        route,
        `interval`,
        quantity,
        refills,
        start_date,
        end_date,
        indication,
        provider_id,
        active,
        note
    FROM prescriptions
    WHERE patient_id = {{pid}}
      AND (encounter = {{enc_id}} OR encounter IS NULL)
    ORDER BY date_added DESC
""",

    # Hospital Course (assessment + plan from SOAP)
    "Hospital_Course": f"""
        SELECT fe.encounter, fe.date, fs.assessment, fs.plan
        FROM form_encounter fe
        LEFT JOIN form_soap fs 
            ON fs.pid = fe.pid 
           AND DATE(fs.date) = DATE(fe.date)
        WHERE fe.pid = {{pid}}
        ORDER BY fe.date DESC
    """,

    "Care_Plan": f"""
        SELECT encounter, date, codetext, description, note_related_to
        FROM form_care_plan
        WHERE pid = {{pid}}
    """,

    "Discharge_Instructions": f"""
        SELECT encounter, instruction, date
        FROM form_clinical_instructions
        WHERE pid = {{pid}}
    """
}


# Fetch encounters
cursor.execute(f"SELECT encounter, date, date_end FROM form_encounter WHERE pid={pid} ORDER BY date DESC")
encounters = cursor.fetchall()

with open("patient_5555_full.txt", "w") as f:

    # 1. Patient-level sections
    f.write("==== Patient-Level Information ====\n")
    for section, query in queries_patient_level.items():
        f.write(f"\n-- {section} --\n")
        try:
            cursor.execute(query.format(pid=pid))
            rows = cursor.fetchall()
            if rows:
                for row in rows:
                    f.write(format_row(section, row) + "\n")
            else:
                f.write("No data found.\n")
        except mysql.connector.Error as e:
            f.write(f"Error in {section}: {e}\n")

    # 2. Encounter-level sections
    cursor.execute(f"SELECT encounter, date, date_end FROM form_encounter WHERE pid={pid} ORDER BY date DESC LIMIT 1")
    encounters = cursor.fetchall()

    for enc in encounters:
        enc_id = enc['encounter']
        start_date = enc.get('date')
        end_date = enc.get('date_end')

        f.write(f"\n\n==== Discharge Summary for Encounter {enc_id} ({start_date} → {end_date}) ====\n")

        for section, query in queries_per_encounter.items():
            f.write(f"\n-- {section} --\n")
            try:
                cursor.execute(query.format(pid=pid, enc_id=enc_id))
                rows = cursor.fetchall()

                # Filter to this encounter if encounter column exists
                if rows and 'encounter' in rows[0]:
                    # rows = [r for r in rows if r['encounter'] == enc_id] # Has to be Encounter Specific
                    rows = [r for r in rows if (r['encounter'] == enc_id or not r['encounter'])] # either match the encounter or are general (NULL)
                if rows:
                    for row in rows:
                        f.write(format_row(section, row) + "\n")
                else:
                    f.write("No data found.\n")
            except mysql.connector.Error as e:
                f.write(f"Error in {section}: {e}\n")

cursor.close()
conn.close()
print("Extraction complete → patient_5555_full.txt")



# Questions:
# 1) So many dates in prescription table, which one to take?