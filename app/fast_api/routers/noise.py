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
    tmp_path = None

    if not body.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="audio 파일만 업로드 가능합니다.")

    try:
        # 임시 파일로 저장
        suffix = os.path.splitext(body.filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file_bytes = await body.read()
            tmp.write(file_bytes)
            tmp_path = tmp.name

        logger.info("[NOISE] saved temp file: %s", tmp_path)

        # 분석
        result = analyze_wall_material_api(tmp_path)

    except ValueError as e:
        logger.warning("[NOISE] invalid audio input: %s", e, exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        # 여기서 반드시 as e 붙이거나, e를 쓰지 말고 logger.exception만 써야 함
        logger.exception("[NOISE] unexpected error in analyze_noise: %s", e)
        raise HTTPException(status_code=500, detail="오디오 분석 중 오류가 발생했습니다.")

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                logger.info("[NOISE] removed temp file: %s", tmp_path)
            except Exception:
                logger.warning("[NOISE] failed to remove temp file: %s", tmp_path, exc_info=True)

    return {
        "code": "COMMON200",
        "message": "성공입니다.",
        "result": {
            "grade": result["grade"],
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
