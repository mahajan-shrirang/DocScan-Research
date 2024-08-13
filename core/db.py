import sqlite3
from datetime import datetime
import pandas as pd
import uuid


def get_db_conn():
    conn = sqlite3.connect("data.db")
    return conn


def dump_data_to_db(data: pd.DataFrame):
    conn = get_db_conn()
    data.to_sql("data", conn, if_exists="replace", index=False)
    conn.close()


def get_data_from_db(query: str):
    conn = get_db_conn()
    data = pd.read_sql(query, conn)
    conn.close()
    return data


def get_columns():
    conn = get_db_conn()
    c = conn.cursor()
    c.execute("PRAGMA table_info(data)")
    columns = c.fetchall()
    conn.close()
    return [column[1] for column in columns]


def create_processes_table(conn_processes):
    mycursor = conn_processes.cursor()
    mycursor.execute("""CREATE TABLE IF NOT EXISTS processes (
    id INTEGER PRIMARY KEY,
    job_id TEXT,
    process_id TEXT,
    status TEXT,
    file_path TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
    """)
    return mycursor


def update_process_status(job_id, status):
    conn_processes = get_db_conn()
    mycursor = conn_processes.cursor()
    mycursor.execute("""UPDATE processes SET status = ? WHERE job_id = ?""", (status, job_id))
    conn_processes.commit()
    mycursor.close()
    conn_processes.close()


def update_process_id(job_id, process_id):
    conn_processes = get_db_conn()
    mycursor = conn_processes.cursor()
    mycursor.execute("""UPDATE processes SET process_id = ? WHERE job_id = ?""", (process_id, job_id))
    conn_processes.commit()
    mycursor.close()
    conn_processes.close()


def insert_process_details_into_db(process_id, file_path) -> str:
    conn_processes = get_db_conn()
    mycursor = conn_processes.cursor()
    currentDateTime = datetime.now()
    job_id = str(uuid.uuid4())
    mycursor.execute(
        """INSERT INTO processes(job_id, process_id, status, file_path, timestamp) VALUES(?,?,?,?,?)""",
        (job_id, process_id, "In Queue", file_path, currentDateTime),
    )
    conn_processes.commit()
    mycursor.close()
    conn_processes.close()
    return job_id


def get_process_details(job_id):
    conn_processes = get_db_conn()
    mycursor = conn_processes.cursor()
    mycursor.execute("SELECT * FROM processes WHERE job_id = ?", (job_id,))
    rows = mycursor.fetchall()
    conn_processes.commit()
    mycursor.close()
    conn_processes.close()
    process = {
        "job_id": rows[0][1],
        "process_id": rows[0][2],
        "status": rows[0][3],
        "file_path": rows[0][4],
        "timestamp": rows[0][5],
    }
    return process


def get_processes():
    conn_processes = get_db_conn()
    mycursor = conn_processes.cursor()
    mycursor.execute("SELECT * FROM processes")
    rows = mycursor.fetchall()
    conn_processes.commit()
    mycursor.close()
    conn_processes.close()
    df = pd.DataFrame(rows[-10:], columns=["ID", "job_id", "process_id", "status", "file_path", "Timestamp"])
    return df
