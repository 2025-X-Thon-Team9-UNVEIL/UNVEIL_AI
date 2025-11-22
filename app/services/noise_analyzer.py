# app/services/noise_analyzer.py

"""
벽체 재질 분석 모듈
test.ipynb의 분석 로직을 API용으로 변환
"""

import numpy as np
from scipy import stats, signal
import logging
import soundfile as sf
logger = logging.getLogger("uvicorn")


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Butterworth Band-pass Filter를 사용하여 특정 주파수 대역만 추출합니다.
    """
    logger.info(f"[NOISE] bandpass filter start: {lowcut}~{highcut}Hz")
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype="band")
    y = signal.filtfilt(b, a, data)
    logger.info(f"[NOISE] bandpass filter end: {lowcut}~{highcut}Hz")
    return y


def get_decay_curve_and_rt60(y, sr):
    """
    슈뢰더 역방향 적분을 통해 에너지 감쇠 곡선을 추출하고, RT60을 계산합니다.
    """
    logger.info("[NOISE] RT60 calculation start (len=%d, sr=%d)", len(y), sr)

    # 피크 지점 찾기
    peak_index = np.argmax(np.abs(y))
    y = y[peak_index:]

    # 에너지 계산
    energy = y**2

    # 슈뢰더 역방향 적분
    s_energy = np.flip(np.cumsum(np.flip(energy)))

    # dB 변환
    epsilon = 1e-10
    s_db = 10 * np.log10(s_energy / np.max(s_energy) + epsilon)

    # 선형 구간 추출 (-5dB ~ -25dB)
    idx_start = np.where(s_db <= -5)[0]
    idx_end = np.where(s_db <= -25)[0]

    # 기본값 미리 설정해 두고, 실패 케이스에서는 이 값으로 리턴
    default_rt60 = 0.1

    if len(idx_start) == 0 or len(idx_end) == 0:
        logger.warning("[NOISE] RT60 구간 탐색 실패 (idx_start/idx_end 없음), rt60=%.3f 사용", default_rt60)
        return default_rt60, s_db

    idx_start = idx_start[0]
    idx_end = idx_end[0]

    if idx_start >= idx_end:
        logger.warning("[NOISE] RT60 구간이 잘못됨 (idx_start >= idx_end), rt60=%.3f 사용", default_rt60)
        return default_rt60, s_db

    # 선형 회귀 분석
    x = np.arange(idx_start, idx_end)
    y_slice = s_db[idx_start:idx_end]

    slope, _, _, _, _ = stats.linregress(x, y_slice)

    if slope == 0:
        slope = -0.001

    # RT60 계산
    rt60 = -60 / slope / sr
    rt60 = abs(rt60)

    logger.info("[NOISE] RT60 calculation end -> %.4f", rt60)
    return rt60, y_slice


def _grade_code_to_letter(grade_code: str) -> str:
    """
    SAFE       -> A
    NORMAL     -> B
    WARNING    -> C
    REFLECTIVE -> C
    UNKNOWN    -> C
    """
    mapping = {
        "SAFE": "A",
        "NORMAL": "B",
        "WARNING": "C",
        "REFLECTIVE": "C",
        "UNKNOWN": "C",
    }
    return mapping.get(grade_code, "C")


def analyze_wall_material_api(file_path: str) -> dict:
    """
    오디오 파일을 분석하여 벽체 재질 등급을 반환합니다.

    [파라미터]
    - file_path: 분석할 오디오 파일 경로 (.wav 등)

    [반환값]
    - dict: grade, grade_code, bass_ratio, rt60 값들 포함
    """
    logger.info("[NOISE] start analyze_wall_material_api, file=%s", file_path)

    # 1) 오디오 로딩: soundfile만 사용
    try:
        logger.info("[NOISE] loading audio via soundfile...")
        y, sr = sf.read(file_path, always_2d=False)

        # 스테레오인 경우 → 모노로 평균
        if y.ndim > 1:
            logger.info("[NOISE] input audio has %d channels, converting to mono", y.shape[1])
            y = y.mean(axis=1)

        logger.info("[NOISE] loaded audio via soundfile, sr=%s, len=%s", sr, len(y))
    except Exception as e:
        logger.error("[NOISE] failed to load audio via soundfile: %s", e)
        raise ValueError(f"오디오 파일을 로드할 수 없습니다: {str(e)}")

    # 2) 주파수 대역 분해
    logger.info("[NOISE] filtering low-band...")
    y_low = butter_bandpass_filter(y, 125, 500, sr, order=3)
    logger.info("[NOISE] filtering high-band...")
    y_high = butter_bandpass_filter(y, 1000, 4000, sr, order=5)

    # 3) RT60 계산
    logger.info("[NOISE] computing RT60 values...")
    rt60_full, _ = get_decay_curve_and_rt60(y, sr)
    rt60_low, _ = get_decay_curve_and_rt60(y_low, sr)
    rt60_high, _ = get_decay_curve_and_rt60(y_high, sr)

    # 4) Bass Ratio 계산
    bass_ratio = rt60_low / (rt60_high + 1e-5)

    logger.info(
        "[NOISE] RT60 full=%.3f, low=%.3f, high=%.3f, bass_ratio=%.3f",
        rt60_full, rt60_low, rt60_high, bass_ratio,
    )

    # 5) 벽체 재질 판별
    grade = "판단 보류"
    grade_code = "UNKNOWN"

    # (A) 전체 RT60이 매우 짧으면 -> 흡음 환경
    if rt60_full < 0.3:
        grade = "흡음 환경 (Safe)"
        grade_code = "SAFE"

    # (B) Bass ratio가 0.1 ~ 0.3이면 -> 일반 콘크리트/조적벽
    elif 0.1 <= bass_ratio <= 0.3:
        grade = "콘크리트/조적벽 (Normal)"
        grade_code = "NORMAL"

    # (C) Bass ratio가 0.3 이상이면 -> 가벽/중공벽 의심
    elif bass_ratio > 0.3:
        grade = "가벽/중공벽 의심 (Warning)"
        grade_code = "WARNING"

    # (D) 그 외 -> 반사성 표면
    else:
        grade = "반사성 표면 (Glass/Tile)"
        grade_code = "REFLECTIVE"

    grade_letter = _grade_code_to_letter(grade_code)

    logger.info(
        "[NOISE] grade_code=%s, grade_letter=%s, desc=%s",
        grade_code, grade_letter, grade,
    )

    # -----------------------------------------------
    # -----------------------------------------------
    logger.info("-" * 50)
    logger.info(f"📢 분석 결과: {grade}")
    logger.info(f"📊 Bass Ratio: {bass_ratio:.2f}")
    logger.info(f"   - 저주파 RT60: {rt60_low:.2f}초")
    logger.info(f"   - 고주파 RT60: {rt60_high:.2f}초")
    logger.info(f"   - 전체 RT60: {rt60_full:.2f}초")

    return {
        "grade": grade_letter,          # API 스펙: A ~ C
        "grade_desc": grade,            # 내부적으로 쓰거나 디버깅용
        "grade_code": grade_code,
        "bass_ratio": round(bass_ratio, 2),
        "rt60_full": round(rt60_full, 2),
        "rt60_low": round(rt60_low, 2),
        "rt60_high": round(rt60_high, 2),
    }