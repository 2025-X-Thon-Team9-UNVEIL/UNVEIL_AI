# app/services/noise_analyzer.py

"""
벽체 재질 분석 모듈
test.ipynb의 분석 로직을 API용으로 변환
"""

import numpy as np
from scipy import stats, signal
import logging
logger = logging.getLogger("uvicorn")


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Butterworth Band-pass Filter를 사용하여 특정 주파수 대역만 추출합니다.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype="band")
    y = signal.filtfilt(b, a, data)
    return y


def get_decay_curve_and_rt60(y, sr):
    """
    슈뢰더 역방향 적분을 통해 에너지 감쇠 곡선을 추출하고, RT60을 계산합니다.
    """
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

    if len(idx_start) == 0 or len(idx_end) == 0:
        return 0.1, s_db

    idx_start = idx_start[0]
    idx_end = idx_end[0]

    if idx_start >= idx_end:
        return 0.1, s_db

    # 선형 회귀 분석
    x = np.arange(idx_start, idx_end)
    y_slice = s_db[idx_start:idx_end]

    slope, _, _, _, _ = stats.linregress(x, y_slice)

    if slope == 0:
        slope = -0.001

    # RT60 계산
    rt60 = -60 / slope / sr

    return abs(rt60), y_slice


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
    logger.info(f"[NOISE] start analyze_wall_material_api, file={file_path}")

    import librosa
    logger.info("[NOISE] imported librosa")

    """
    오디오 파일을 분석하여 벽체 재질 등급을 반환합니다.

    [파라미터]
    - file_path: 분석할 오디오 파일 경로 (.wav 등)

    [반환값]
    - dict: grade, grade_code, bass_ratio, rt60 값들 포함
    """
    try:
        # 오디오 파일 로딩
        y, sr = librosa.load(file_path, sr=None)
        logger.info(f"[NOISE] loaded audio, sr={sr}, len={len(y)}")
    except Exception as e:
        raise ValueError(f"오디오 파일을 로드할 수 없습니다: {str(e)}")

    # 주파수 대역 분해
    y_low = butter_bandpass_filter(y, 125, 500, sr, order=5)
    y_high = butter_bandpass_filter(y, 1000, 4000, sr, order=5)

    # 대역별 RT60 계산
    rt60_full, _ = get_decay_curve_and_rt60(y, sr)
    rt60_low, _ = get_decay_curve_and_rt60(y_low, sr)
    rt60_high, _ = get_decay_curve_and_rt60(y_high, sr)

    # Bass Ratio 계산
    bass_ratio = rt60_low / (rt60_high + 1e-5)

    # 디버깅 출력 추가
    print("[DEBUG] rt60_full:", rt60_full)
    print("[DEBUG] rt60_low:", rt60_low)
    print("[DEBUG] rt60_high:", rt60_high)
    print("[DEBUG] bass_ratio:", bass_ratio)

    # 벽체 재질 판별
    grade = "판단 보류"
    grade_code = "UNKNOWN"

    if rt60_full < 0.3:
        grade = "흡음 환경 (Safe)"
        grade_code = "SAFE"

    elif 0.1 <= bass_ratio <= 0.3:
        grade = "콘크리트/조적벽 (Normal)"
        grade_code = "NORMAL"

    elif bass_ratio > 0.3:
        grade = "가벽/중공벽 의심 (Warning)"
        grade_code = "WARNING"

    else:
        grade = "반사성 표면 (Glass/Tile)"
        grade_code = "REFLECTIVE"

    grade_letter = _grade_code_to_letter(grade_code)
    print("[DEBUG] grade_code:", grade_code)
    print("[DEBUG] grade_letter:", grade_letter)

    return {
        "grade": grade_letter,          # API 스펙: A ~ D
        "grade_desc": grade,            # 내부적으로 쓰거나 디버깅용
        "grade_code": grade_code,
        "bass_ratio": round(bass_ratio, 2),
        "rt60_full": round(rt60_full, 2),
        "rt60_low": round(rt60_low, 2),
        "rt60_high": round(rt60_high, 2),
    }