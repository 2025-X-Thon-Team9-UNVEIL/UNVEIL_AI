# app/fast_api/main.py
from fastapi import FastAPI
import uvicorn
import os
from app.core.database import get_mysql_connection

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "UNVEIL AI FastAPI alive!"}

@app.get("/test-db")
async def test_db():
    conn = get_mysql_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS mytest (id INTEGER PRIMARY KEY)")
        cursor.execute("INSERT INTO mytest (id) VALUES (1)")
        cursor.execute("SELECT * FROM mytest")
        rows = cursor.fetchall()
        return rows
    finally:
        conn.close()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app.fast_api.main:app", host="0.0.0.0", port=port, reload=False)

# uvicorn app.fast_api.main:app --reload
