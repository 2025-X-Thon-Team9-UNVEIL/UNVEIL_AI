# app/fast_api/routers/noise.py

import os
import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.noise_analyzer import analyze_wall_material_api
import logging
import time

router = APIRouter(prefix="/api", tags=["noise"])
logger = logging.getLogger("uvicorn")

@router.post("/noise")
async def analyze_noise(body: UploadFile = File(...)):
    logger.info(f"/api/noise hit, filename={body.filename}")
    start = time.time()
    """
    FormData로 음성파일(body)을 받아 벽체 재질 등급을 분석하는 API

    Request (multipart/form-data):
        - body: 음성 파일(.wav, .mp3, .m4a 등)

    Response:
    {
        "code": "COMMON200",
        "message": "성공입니다.",
        "result": {
            "grade": "A"
        },
        "success": true
    }
    """
    # 파일 타입 간단 검증 (선택)
    if not body.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="audio 파일만 업로드 가능합니다.")

    # 업로드된 파일을 임시 파일로 저장
    try:
        suffix = os.path.splitext(body.filename)[1] or ".wav"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file_bytes = await body.read()
            tmp.write(file_bytes)
            tmp_path = tmp.name

        logger.info("[NOISE] before analyze_wall_material_api")   # 분석 전
        result = analyze_wall_material_api(tmp_path)
        logger.info("[NOISE] after analyze_wall_material_api, result=%s", result)  # 분석 후

        # 분석 로직 호출
        result = analyze_wall_material_api(tmp_path)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="오디오 분석 중 오류가 발생했습니다.")
    finally:
        # 임시 파일 정리
        try:
            if "tmp_path" in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
    
    elapsed = time.time() - start  # 총 소요 시간
    logger.info(f"[NOISE] /api/noise done in {elapsed:.2f} seconds")

    # API 명세서 포맷에 맞게 응답 래핑
    return {
        "code": "COMMON200",
        "message": "성공입니다.",
        "result": {
            "grade": result["grade"]
        },
        "success": True,
    }

@router.post("/noise-test")
async def noise_test():
    logger.info("/api/noise-test hit")
    return {
        "code": "COMMON200",
        "message": "테스트 성공입니다.",
        "result": {"grade": "B"},
        "success": True,
    }
