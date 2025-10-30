import mysql.connector

conn = mysql.connector.connect(
    host="127.0.0.1",
    user="openemr",
    password="openemr",
    database="openemr"
)
cursor = conn.cursor()

cursor.execute("SHOW TABLES")
tables = cursor.fetchall()

if tables:
    print("Tables in the database:\n")
    for table in tables:
        table_name = table[0]

        try:
            cursor.execute(f"SHOW COLUMNS FROM `{table_name}`")
            cols = [col[0] for col in cursor.fetchall()]
            print(f"{table_name} → Columns: {', '.join(cols)}")
        except mysql.connector.Error as e:
            print(f"{table_name} → Error retrieving columns: {e}")
else:
    print("No tables found.")

cursor.close()
conn.close()
