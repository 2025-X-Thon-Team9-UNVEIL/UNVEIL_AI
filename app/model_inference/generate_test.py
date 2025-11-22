import numpy as np
import soundfile as sf
import os

# --- 설정 ---
SR = 44100  # 샘플링 레이트 (고음질)
DURATION = 4.0 # 전체 길이 4초
CLAP_POS = 0.5 # 0.5초 시점에 박수 소리

def create_synthetic_reverb(rt60_target, filename):
    """
    목표 RT60 값을 가진 합성 오디오 파일을 생성합니다.
    원리: 화이트 노이즈에 지수적으로 감소하는 포락선(Envelope)을 곱합니다.
    """
    length = int(SR * DURATION)
    t = np.linspace(0, DURATION, length)
    
    # 1. 기본 배경 (완전 무음)
    signal = np.zeros(length)
    
    # 2. 박수 소리 위치 설정 (시작 인덱스)
    start_idx = int(SR * CLAP_POS)
    
    # 3. 잔향 생성 (White Noise)
    remaining_len = length - start_idx
    noise = np.random.randn(remaining_len) * 0.5 # 0.5는 볼륨 조절
    
    # 4. 감쇠 곡선 적용 (Exponential Decay)
    # RT60 공식에 따라 감쇠 계수 계산
    if rt60_target > 0:
        decay_coeff = 6.91 / rt60_target # T60 = 6.91 / lambda
        decay_envelope = np.exp(-decay_coeff * t[:remaining_len])
    else:
        # 잔향이 0인 경우 (그냥 아주 짧은 충격음만 남김)
        decay_envelope = np.zeros(remaining_len)
        decay_envelope[:500] = np.linspace(1, 0, 500) # 500샘플(약 0.01초)만에 사라짐

    # 노이즈에 감쇠 적용
    reverb_tail = noise * decay_envelope
    
    # 5. 신호 합치기
    signal[start_idx:] = reverb_tail
    
    # 6. 파일 저장 (16bit PCM WAV)
    sf.write(filename, signal, SR, subtype='PCM_16')
    print(f"✅ 파일 생성 완료: {filename} (목표 RT60: {rt60_target}초)")

# --- 실행 ---
if __name__ == "__main__":
    print("테스트 오디오 파일 생성을 시작합니다...")
    
    # 1. 좋은 방 (잔향 0.3초 - 거의 흡음됨)
    create_synthetic_reverb(rt60_target=0.3, filename="test_good_room.wav")
    
    # 2. 보통 방 (잔향 0.6초 - 일반적)
    create_synthetic_reverb(rt60_target=0.6, filename="test_normal_room.wav")
    
    # 3. 최악의 방 (잔향 1.5초 - 동굴 수준, 매우 울림)
    create_synthetic_reverb(rt60_target=1.5, filename="test_bad_room.wav")

    print("\n모든 파일이 생성되었습니다. 분석 코드에 넣고 돌려보세요!")