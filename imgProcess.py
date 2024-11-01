import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import time
import os

# 이미지 크기 상수
INPUT_SIZE = 320

def load_img(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img = img.convert('RGB')
    
    # MobileNet에 최적화된 크기로 조정
    img = img.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.LANCZOS)
    
    return np.array(img)

def run_detector(detector, img_path):
    img = load_img(img_path)
    converted_img = tf.cast(img, tf.uint8)[tf.newaxis, ...]  # uint8 형식으로 변환

    start_time = time.time()
    
    @tf.function(experimental_relax_shapes=True)
    def detect(image):
        return detector(image)
    
    # 입력 텐서만 detector에 전달
    result = detect(converted_img)
    end_time = time.time()

    # SSD MobileNet의 출력 키 이름 변경
    result = {
        'detection_boxes': result['detection_boxes'],
        'detection_scores': result['detection_scores'],
        'detection_classes': result['detection_classes']
    }
    
    # 클래스 이름 매핑 (COCO 데이터셋 기준)
    coco_classes = {
        17: 'cat',  # COCO 데이터셋에서 고양이는 17번
        # 필요한 경우 다른 클래스 추가
    }
    
    # 클래스 엔티티 생성
    result['detection_class_entities'] = np.array([
        coco_classes.get(int(class_id), f'class_{class_id}').encode('utf-8')
        for class_id in result['detection_classes'][0]
    ])

    print(f"Found {len(result['detection_scores'][0])} objects.")
    print(f"Inference time: {end_time - start_time:.2f} seconds")

    return img, result



def draw_boxes(image, result, min_score=0.5):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    original_size = image.size
    image = image.resize(original_size, Image.Resampling.LANCZOS)
    
    draw = ImageDraw.Draw(image)
    font_size = max(int(image.width / 40), 12)
    line_width = max(int(image.width / 200), 2)
    
    try:
        font = ImageFont.truetype('arial.ttf', size=font_size)
    except IOError:
        font = ImageFont.load_default()
    
    # SSD MobileNet 출력 형식에 맞게 수정
    scores = result['detection_scores'][0]
    boxes = result['detection_boxes'][0]
    classes = result['detection_class_entities']
    
    valid_detections = [
        (score, box, class_entity)
        for score, box, class_entity in zip(scores, boxes, classes)
        if score > min_score
    ]
    
    for score, box, class_entity in valid_detections:
        ymin, xmin, ymax, xmax = box
        left = xmin * image.width
        right = xmax * image.width
        top = ymin * image.height
        bottom = ymax * image.height
        
        draw.rectangle((left, top, right, bottom), 
                      outline='red', 
                      width=line_width)
        label = f"{class_entity.decode('utf-8')}: {score:.2f}"
        draw.text((left, top), label, font=font, fill='red')
    
    return image

def prc(imgPath):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print("GPU is configured for dynamic memory growth")
        except RuntimeError as e:
            print(e)
    
    try:
        # SSD MobileNet v2 모델 로드 (320x320)
        module_handle = "https://www.kaggle.com/models/tensorflow/ssd-mobilenet-v2/TensorFlow2/fpnlite-320x320/1"
        print("Loading model...")
        detector = hub.load(module_handle)
        print("Model loaded.")

        img, result = run_detector(detector, imgPath)
        
        # SSD MobileNet 출력 형식에 맞게 수정
        scores = result['detection_scores'][0]
        classes = result['detection_class_entities']
        
        detected_info = [
            (class_entity.decode('utf-8').lower(), f"{score:.2f}")
            for score, class_entity in zip(scores, classes)
            if score > 0.5
        ]
        
        img_with_boxes = draw_boxes(img, result)
        img_with_boxes = Image.fromarray(img_with_boxes) if isinstance(img_with_boxes, np.ndarray) else img_with_boxes
        img_with_boxes.save(imgPath, quality=95, optimize=True)
        
        cat_detected = any('cat' in cls for cls, _ in detected_info)
        result_items = [f"{cls}: {sc}" for cls, sc in detected_info]
        print('Success' if cat_detected else 'Fail', ''.join(result_items))
        return f"{'Success' if cat_detected else 'Fail'}, {', '.join(result_items)}"

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"

if __name__ == "__main__":
    try:
        imgPath = os.path.join('temp', "n.jpg")
        result = prc(imgPath=imgPath)
        print(result)
    except Exception as e:
        print(f"Program execution error: {str(e)}")
        import traceback
        traceback.print_exc()