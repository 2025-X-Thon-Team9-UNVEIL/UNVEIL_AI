# app/fast_api/main.py
from fastapi import FastAPI
from app.core.database import get_mysql_connection
from app.fast_api.routers.noise import router as noise_router

app = FastAPI()


# 라우터 등록
app.include_router(noise_router)

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

# uvicorn app.fast_api.main:app --reload