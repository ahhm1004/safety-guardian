# 행동 패턴 분석 시스템 (Behavior Pattern Analysis System)

이 프로젝트는 영상 스트림에서 사람의 행동 패턴을 분석하는 시스템입니다. 객체 검출, 자세 추정, 행동 인식을 순차적으로 적용하여 관찰 대상의 행동을 자동으로 분류합니다.

## 시스템 구성

이 시스템은 다음과 같은 단계로 구성됩니다:

1. **영상 입력**: 다양한 소스(비디오 파일, 카메라, RTSP 스트림)에서 영상 프레임 획득
2. **객체 검출**: YOLO 모델을 사용하여 프레임 내 사람 객체 검출 및 바운딩 박스 추출
3. **자세 추정**: 검출된 각 사람에 대해 관절 좌표(키포인트) 추출
4. **행동 인식**: 자세 정보를 바탕으로 행동 패턴 분류 (걷기, 앉기, 넘어짐, 싸움 등)
5. **결과 시각화 및 알림**: 분석 결과를 화면에 표시하고 특정 행동 발생 시 알림 생성

## 설치 방법

### 요구 사항
- Python 3.8 이상
- CUDA 지원 GPU (권장, CPU에서도 실행 가능)

### 설치 과정
1. 저장소 복제
```bash
git clone https://github.com/username/behavior-analysis.git
cd behavior-analysis
```

2. 의존성 설치
```bash
pip install -r requirements.txt
```

3. AWS S3 설정 (선택 사항)
```bash
# config.py 파일 내 AWS 접근 키 설정
```

## 사용 방법

### 기본 실행
```bash
python main.py
```

### 옵션 지정 실행
```bash
# 위 명령어를 실행하면 현재 설정된 영상 소스의 첫 프레임에서 위험 구역을 마우스로 선택할 수 있습니다. 다음 옵션도 사용 가능합니다:
python scripts/danger_zone_selector.py
python scripts/danger_zone_selector.py --image_path C:\workspace\python\vision\data\New_Sample\raw\service\ws\p1\L-210916_G03_D_WS-09_001_0001.jpg --save
python scripts/danger_zone_selector.py --image_path C:\workspace\python\vision\data/test.jpg

# 비디오 파일 입력
python main.py --input .\data\test.mp4 --type file --display --output results/test.mp4

# 웹캠 입력
python main.py --input 0 --type camera

# RTSP 스트림 입력
python main.py --input rtsp://example.com/stream --type rtsp


# 이미지 폴더 입력 (프레임으로 저장된 이미지 사용)
python main.py --input data/frames --type image_folder --image_pattern *.jpg

# 로컬 이미지 폴더에서 실행
python main.py --input .\data\New_Sample\raw\service\ws/p1 --type image_folder --image_pattern *.jpg --display --output results/my_analysis.mp4

# 화면 출력 활성화
python main.py --display

# 모델 지정
python main.py --detector yolov8 --pose_estimator mediapipe --action_recognizer rule
```

## 이미지 폴더 데이터셋 사용

이미지 폴더를 사용하면 프레임 단위로 미리 저장된 이미지 시퀀스를 분석할 수 있습니다. 이 기능은 다음과 같은 상황에서 유용합니다:

1. 비디오 디코딩 과정을 건너뛰어 성능 향상
2. 미리 전처리된 프레임 사용
3. 다양한 소스에서 수집된 이미지 분석
4. 복잡한 데이터셋 구조 처리

### 이미지 폴더 구조
```
data/frames/
  ├── frame_0001.jpg
  ├── frame_0002.jpg
  ├── frame_0003.jpg
  └── ...
```

### 이미지 폴더 옵션
- `--type image_folder`: 입력 소스를 이미지 폴더로 지정
- `--input [경로]`: 이미지가 저장된 폴더 경로
- `--image_pattern [패턴]`: 이미지 파일 패턴 (*.jpg, frame_*.png 등)

## 폴더 구조

```
.
├── config.py               # 시스템 설정 파일
├── main.py                 # 메인 스크립트
├── requirements.txt        # 의존성 목록
├── data/                   # 데이터 디렉토리
├── models/                 # 모델 파일 디렉토리
├── results/                # 결과 저장 디렉토리
├── utils/                  # 유틸리티 함수
│   └── s3_utils.py         # S3 관련 유틸리티
└── modules/                # 핵심 모듈
    ├── video_input/        # 비디오 입력 모듈
    ├── object_detection/   # 객체 검출 모듈
    ├── pose_estimation/    # 자세 추정 모듈
    ├── action_recognition/ # 행동 인식 모듈
    └── visualization/      # 결과 시각화 모듈
```

## 모듈별 설명

### 1. 비디오 입력 모듈
다양한 입력 소스에서 영상을 읽어오는 기능을 제공합니다.
- 파일, 카메라, RTSP 스트림 지원
- 프레임 전처리 및 변환

### 2. 객체 검출 모듈
YOLO 모델을 사용하여 프레임 내 사람을 검출합니다.
- YOLOv5, YOLOv8 지원
- 설정 가능한 검출 임계값

### 3. 자세 추정 모듈
검출된 사람의 관절 좌표를 추정합니다.
- Mediapipe Pose 모델 사용
- 키포인트 및 연결 정보 추출

### 4. 행동 인식 모듈
자세 정보를 바탕으로 행동을 분류합니다.
- 규칙 기반 분류 (넘어짐, 서있음, 앉음 등)
- LSTM 기반 시퀀스 분류 (설정 시)

### 5. 시각화 모듈
분석 결과를 시각적으로 표현합니다.
- 바운딩 박스, 골격, 행동 라벨 표시
- 결과 비디오 저장 및 알림 처리

## 커스터마이징

시스템의 다양한 측면을 `config.py` 파일을 통해 쉽게 커스터마이징할 수 있습니다.

### 설정 가능 항목
- 입력 소스 및 해상도
- 객체 검출 모델 및 임계값
- 자세 추정 옵션
- 행동 인식 방법 및 클래스
- 시각화 및 알림 설정
- S3 설정

## 추가 자료

- [데이터셋 가이드](data/README.md)
- [모델 가이드](models/README.md)

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 