"""
객체 검출(Object Detection) 모듈

YOLO 모델을 사용하여 이미지에서 사람을 검출하는 기능을 제공합니다.
"""

import os
import sys
import cv2
import numpy as np
import logging
import torch
from abc import ABC, abstractmethod

# 상위 디렉토리 추가해서 config.py 접근 가능하게 함
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import DETECTION_CONFIG

logger = logging.getLogger(__name__)

class ObjectDetector(ABC):
    """객체 검출기 추상 클래스"""
    
    @abstractmethod
    def load_model(self):
        """모델 로드"""
        pass
        
    @abstractmethod
    def detect(self, frame):
        """
        이미지에서 객체 검출
        
        Args:
            frame (numpy.ndarray): 입력 이미지
            
        Returns:
            list: 검출된 객체 정보 리스트 [x1, y1, x2, y2, confidence, class_id]
        """
        pass


class YOLODetector(ObjectDetector):
    """YOLO 모델을 사용한 객체 검출기"""
    
    def __init__(self, model_path=None, confidence_threshold=0.5, 
                 nms_threshold=0.45, device='cuda:0', classes=None):
        """
        Args:
            model_path (str): 모델 파일 경로 (None이면 config에서 가져옴)
            confidence_threshold (float): 객체 검출 임계값
            nms_threshold (float): NMS(Non-Maximum Suppression) 임계값
            device (str): 연산 장치 ('cpu' 또는 'cuda:0', 'cuda:1' 등)
            classes (list): 검출할 클래스 ID 목록 (None이면 모든 클래스)
        """
        self.model_path = model_path or DETECTION_CONFIG['model_path']
        self.confidence_threshold = confidence_threshold or DETECTION_CONFIG['confidence_threshold']
        self.nms_threshold = nms_threshold or DETECTION_CONFIG['nms_threshold']
        self.device = device or DETECTION_CONFIG['device']
        self.classes = classes or DETECTION_CONFIG['classes']
        self.model = None
        self.model_type = DETECTION_CONFIG['model_type']
        
        # 하드웨어 가속 설정 추가
        self.use_tensorrt = DETECTION_CONFIG.get('use_tensorrt', False) if 'cuda' in str(self.device) else False
        self.use_half_precision = DETECTION_CONFIG.get('use_half_precision', False) if 'cuda' in str(self.device) else False
        
    def load_model(self):
        """
        YOLO 모델 로드
        
        Returns:
            bool: 모델 로드 성공 여부
        """
        try:
            logger.info(f"YOLO 모델 로드 중: {self.model_path}")
            
            # YOLOv5 또는 YOLOv8 모델 로드
            if self.model_type.lower() == 'yolov5':
                # YOLOv5는 직접 pip 설치하지 않고 로컬에서 임포트할 수도 있음
                try:
                    import torch
                    self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                               path=self.model_path, device=self.device)
                    self.model.conf = self.confidence_threshold  # 신뢰도 임계값 설정
                    self.model.iou = self.nms_threshold  # IoU 임계값 설정
                    logger.info("YOLOv5 모델 로드 성공")
                except Exception as e:
                    logger.error(f"YOLOv5 모델 로드 실패: {e}")
                    return False
                    
            elif self.model_type.lower() == 'yolov8':
                try:
                    from ultralytics import YOLO
                    self.model = YOLO(self.model_path)
                    logger.info("YOLOv8 모델 로드 성공")
                except Exception as e:
                    logger.error(f"YOLOv8 모델 로드 실패: {e}")
                    return False
                
            elif self.model_type.lower() == 'yolov11':
                try:
                    from ultralytics import YOLO

                    # 우선 로드할 경로 설정 (.engine -> .onnx -> .pt)
                    pt_path = self.model_path
                    engine_path = pt_path.replace('.pt', '.engine')
                    onnx_path = pt_path.replace('.pt', '.onnx')

                    # 1) TensorRT 엔진 우선 시도
                    if self.use_tensorrt and os.path.exists(engine_path):
                        logger.info(f"TensorRT Detector 엔진 로드 시도: {engine_path}")
                        self.model = YOLO(engine_path)
                        logger.info("TensorRT Detector 엔진 로드 성공")
                        return True

                    # 2) ONNX 모델 시도
                    if os.path.exists(onnx_path):
                        logger.info(f"ONNX Detector 모델 로드 시도: {onnx_path}")
                        self.model = YOLO(onnx_path)
                        logger.info("ONNX Detector 모델 로드 성공")
                        return True

                    # 3) 로컬 PT 모델 시도
                    if os.path.exists(pt_path):
                        logger.info(f"로컬 PT Detector 모델 로드: {pt_path}")
                        self.model = YOLO(pt_path)
                        logger.info("PT Detector 모델 로드 성공")

                        # TensorRT 엔진 자동 변환 시도
                        if self.use_tensorrt and 'cuda' in str(self.device):
                            tensorrt_path = engine_path
                            if not os.path.exists(tensorrt_path):
                                logger.info("TensorRT 엔진 파일이 없어 변환을 시도합니다...")
                                original_cwd = os.getcwd()
                                try:
                                    os.chdir(os.path.dirname(pt_path))
                                    pt_model_for_export = YOLO(os.path.basename(pt_path))
                                    img_size = DETECTION_CONFIG.get('imgsz', 640)
                                    pt_model_for_export.export(format="engine", half=self.use_half_precision, imgsz=img_size, device=self.device)

                                    exported_engine = os.path.join(os.getcwd(), os.path.basename(tensorrt_path))
                                    if os.path.exists(exported_engine) and not os.path.samefile(exported_engine, tensorrt_path):
                                        import shutil
                                        shutil.move(exported_engine, tensorrt_path)
                                        logger.info(f"생성된 TensorRT 엔진을 {tensorrt_path}로 이동했습니다.")
                                except Exception as e:
                                    logger.error(f"TensorRT 변환 실패: {e}")
                                finally:
                                    os.chdir(original_cwd)

                            # 변환 후 로드 시도
                            if os.path.exists(tensorrt_path):
                                try:
                                    self.model = YOLO(tensorrt_path)
                                    logger.info("TensorRT Detector 엔진 로드 성공 (변환 후)")
                                    return True
                                except Exception as e:
                                    logger.warning(f"변환된 TensorRT 엔진 로드 실패, PT 모델 사용 지속: {e}")

                        return True
                    
                except Exception as e:
                    logger.error(f"YOLOv11 모델 로드 실패: {e}")
                    return False
                    
            else:
                logger.error(f"지원하지 않는 YOLO 모델 유형: {self.model_type}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"모델 로드 중 예외 발생: {e}")
            return False
            
    def detect(self, frame):
        """
        이미지에서 객체(사람) 검출
        
        Args:
            frame (numpy.ndarray): 입력 이미지 (BGR 포맷, OpenCV)
            
        Returns:
            list: 검출된 객체 정보 리스트 [x1, y1, x2, y2, confidence, class_id]
        """
        if self.model is None:
            logger.error("모델이 로드되지 않았습니다. load_model()을 먼저 호출하세요.")
            return []
            
        try:
            # RGB로 변환 (YOLO는 RGB 입력 예상)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 모델 타입에 따라 다른 처리 방식 적용
            if self.model_type.lower() == 'yolov5':
                detections = self._detect_yolov5(rgb_frame)
            elif self.model_type.lower() == 'yolov8':
                detections = self._detect_yolov8(rgb_frame)
            elif self.model_type.lower() == 'yolov11':
                detections = self._detect_yolov11(rgb_frame)
            else:
                logger.error(f"지원하지 않는 모델 타입: {self.model_type}")
                return []
            
            # 클래스 필터링 (사람 클래스만 선택)
            if len(detections) > 0 and self.classes:
                detections = self._filter_classes(detections)
                    
            return detections
            
        except Exception as e:
            logger.error(f"객체 검출 중 예외 발생: {e}")
            return []

    def _detect_yolov8(self, rgb_frame):
        """YOLOv8 전용 검출 메서드"""
        results = self.model(rgb_frame, conf=self.confidence_threshold)
        detections = []
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for i in range(len(boxes.xyxy)):
                x1, y1, x2, y2 = boxes.xyxy[i]
                confidence = boxes.conf[i]
                class_id = int(boxes.cls[i])
                detections.append([x1, y1, x2, y2, confidence, class_id])
        return np.array(detections) if detections else np.array([])

    def _detect_yolov11(self, rgb_frame):
        """YOLOv11 전용 검출 메서드 - 최적화된 텐서 처리 방식 사용"""
        try:
            # 임계값 직접 전달 (config에서 읽은 값 사용)
            conf_threshold = float(self.confidence_threshold)
            
            # 감지 실행 - silent=True 제거
            results = self.model(rgb_frame, conf=conf_threshold, verbose=False)
            
            # 결과가 없으면 빈 배열 반환
            if len(results) == 0:
                return np.array([])
                
            if len(results[0].boxes) == 0:
                return np.array([])
            
            # 텐서 처리를 최적화하여 메모리 사용량 감소
            boxes = results[0].boxes
            
            # CPU로 미리 이동하여 불필요한 GPU 메모리 점유 방지
            xyxy = boxes.xyxy.cpu().numpy()
            # conf, cls 텐서는 (N, 1) 형태이므로 1차원으로 평탄화하여 브로드캐스트 오류 방지
            conf = boxes.conf.cpu().numpy().reshape(-1)
            cls = boxes.cls.cpu().numpy().astype(int).reshape(-1)
            
            # NumPy 연산으로 최종 배열 생성
            num_detections = len(xyxy)
            detections = np.zeros((num_detections, 6))
            detections[:, :4] = xyxy
            detections[:, 4] = conf
            detections[:, 5] = cls
            
            # 메모리 명시적 해제
            del results, boxes, xyxy, conf, cls
            
            return detections
        except Exception as e:
            logger.error(f"YOLOv11 감지 중 오류 발생: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return np.array([])

    def _filter_classes(self, detections):
        """클래스 필터링 - 설정된 클래스만 포함"""
        if not isinstance(detections, np.ndarray) or len(detections) == 0:
            return np.array([])
            
        # 클래스 ID가 설정된 클래스 목록에 있는 것만 필터링
        if self.classes is not None:
            # 유효한 클래스 ID만 필터링하는 마스크 생성
            class_ids = detections[:, 5].astype(int)
            mask = np.zeros_like(class_ids, dtype=bool)
            
            # 각 클래스 ID가 허용된 클래스 목록에 있는지 확인
            for class_id in self.classes:
                mask = np.logical_or(mask, class_ids == class_id)
            
            # 필터링 적용
            filtered_detections = detections[mask]
            
            return filtered_detections
        
        # classes가 None인 경우 모든 클래스 반환
        return detections

    def _detect_yolov5(self, rgb_frame):
        """YOLOv5 전용 검출 메서드"""
        results = self.model(rgb_frame)
        return results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, cls]
            
    def draw_detections(self, frame, detections, color=(0, 255, 0), thickness=2):
        """
        검출 결과를 이미지에 그리기
        
        Args:
            frame (numpy.ndarray): 입력 이미지
            detections (list): 검출된 객체 정보 리스트
            color (tuple): 바운딩 박스 색상 (B,G,R)
            thickness (int): 선 두께
            
        Returns:
            numpy.ndarray: 바운딩 박스가 그려진 이미지
        """
        # 검출 결과가 없거나 빈 리스트면 원본 프레임 반환
        if detections is None or (isinstance(detections, np.ndarray) and detections.size == 0) or len(detections) == 0:
            return frame
            
        # NumPy 배열을 리스트로 변환 (필요한 경우)
        det_list = detections.tolist() if isinstance(detections, np.ndarray) else detections
            
        for detection in det_list:
            # 검출 결과 형식 검사
            if len(detection) < 6:
                logger.warning(f"잘못된 검출 결과 형식: {detection}")
                continue
                
            x1, y1, x2, y2, confidence, class_id = detection
            
            # 정수로 변환
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # 바운딩 박스 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # 클래스와 신뢰도 표시 - COCO 클래스 이름 매핑
            class_names = {
                0: "Person", 1: "Bicycle", 2: "Car", 3: "Motorcycle", 4: "Airplane",
                5: "Bus", 6: "Train", 7: "Truck", 8: "Boat", 9: "Traffic Light"
                # 필요시 추가 클래스 추가
            }
            class_name = class_names.get(int(class_id), f"Class {int(class_id)}")
            text = f"{class_name}: {confidence:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, thickness)
                       
        return frame


def create_detector():
    """
    설정에 따라 적절한 객체 검출기를 생성합니다.
    
    Returns:
        ObjectDetector: 생성된 객체 검출기
    """
    model_type = DETECTION_CONFIG['model_type']
    
    if model_type.lower() in ['yolov5', 'yolov8', 'yolov11']:  # YOLOv11 추가
        detector = YOLODetector(
            model_path=DETECTION_CONFIG['model_path'],
            confidence_threshold=DETECTION_CONFIG['confidence_threshold'],
            nms_threshold=DETECTION_CONFIG['nms_threshold'],
            device=DETECTION_CONFIG['device'],
            classes=DETECTION_CONFIG['classes']
        )
        return detector
    else:
        logger.error(f"지원하지 않는 모델 유형: {model_type}")
        return None 