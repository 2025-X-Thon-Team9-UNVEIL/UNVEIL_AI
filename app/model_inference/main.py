import numpy as np
import librosa
from scipy import stats
import matplotlib.pyplot as plt

def calculate_rt60_byme(file_path):
    """
    라이브러리 없이 직접 RT60(잔향 시간)을 계산하는 함수
    """
    # 1. 오디오 로드
    y, sr = librosa.load(file_path, sr=None)

    # 2. 에너지가 가장 큰 지점(박수 소리 피크) 찾기
    # 피크 이전의 잡음은 자르고, 피크부터 분석 시작
    peak_index = np.argmax(np.abs(y))
    y = y[peak_index:]

    # 3. 에너지 감쇠 곡선 (Energy Decay Curve) 만들기
    # 힐베르트 변환 없이 단순 제곱 에너지 사용 (해커톤용 최적화)
    energy = y ** 2
    
    # 슈뢰더 적분 (Schroeder Integration) - 잔향 계산의 표준 공식
    # 뒤에서부터 누적 합을 구해서 에너지 잔량을 계산함
    s_energy = np.flip(np.cumsum(np.flip(energy)))
    
    # 로그 스케일로 변환 (데시벨 dB)
    # 0으로 나누는 에러 방지를 위해 아주 작은 수(epsilon) 더함
    epsilon = 1e-10
    s_db = 10 * np.log10(s_energy / np.max(s_energy) + epsilon)

    # 4. 선형 회귀로 기울기 구하기 (T20 방식 활용)
    # -5dB ~ -25dB 구간의 기울기를 구해서 -60dB까지 걸리는 시간을 추정
    
    # -5dB 지점 찾기
    idx_start = np.where(s_db <= -5)[0]
    if len(idx_start) == 0: return 0.0 # 소리가 너무 작음
    idx_start = idx_start[0]

    # -25dB 지점 찾기
    idx_end = np.where(s_db <= -25)[0]
    if len(idx_end) == 0: 
        # 잔향이 너무 짧아서 -25dB까지 안 떨어지면 끝까지 사용
        idx_end = len(s_db) - 1
    else:
        idx_end = idx_end[0]

    if idx_start >= idx_end:
        return 0.1 # 계산 불가 시 기본값

    # x축(시간), y축(에너지dB) 데이터 준비
    x = np.arange(idx_start, idx_end)
    y_slice = s_db[idx_start:idx_end]

    # 선형 회귀 분석 (기울기 계산)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y_slice)

    # 5. RT60 계산
    # RT60 = -60dB / slope * (1 / 샘플링레이트)
    rt60_value = -60 / slope / sr
    
    plt.figure(figsize=(10, 6))

    # 1. 원본 파형 (파란색)
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y, sr=sr, alpha=0.5)
    plt.title("1. Raw Waveform (Zig-Zag)")

    # 2. 슈뢰더 적분 곡선 (빨간색)
    plt.subplot(2, 1, 2)
    # x축 시간 생성
    times = np.linspace(0, len(y)/sr, len(s_db))
    plt.plot(times, s_db, color='r', linewidth=2)
    plt.title("2. Schroeder Curve (Smooth)")
    plt.grid()

    plt.tight_layout()
    plt.show() # 창 띄우기
    
    return abs(rt60_value)

def analyze_sound(file_path):
    print(f"🔍 분석 중: {file_path}")
    try:
        rt60 = calculate_rt60_byme(file_path)
        
        # 결과 포맷팅
        result = {
            "filename": file_path,
            "rt60_sec": round(rt60, 2),
            "score": 0,
            "risk_level": ""
        }

        # 점수 로직
        if rt60 < 0.4:
            result["score"] = 95
            result["risk_level"] = "Safe (조용함)"
        elif 0.4 <= rt60 <= 0.8:
            result["score"] = 70
            result["risk_level"] = "Normal (보통)"
        else:
            result["score"] = 30
            result["risk_level"] = "Danger (매우 울림)"
            
        return result

    except Exception as e:
        return {"error": str(e)}


# --- 테스트 실행 ---
if __name__ == "__main__":
    # 아까 만든 파일들로 테스트해보세요
    test_files = ["test_good_room.wav", "test_normal_room.wav", "test_bad_room.wav"]
    
    for f in test_files:
        print(analyze_sound(f))