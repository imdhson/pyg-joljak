import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import time
import os

# 이미지 로드 및 전처리 함수
def load_img(path):
    img = Image.open(path)  # PIL Image로 로드
    img = ImageOps.exif_transpose(img)  # EXIF 데이터를 기반으로 이미지 방향 보정
    img = img.convert('RGB')  # RGB 모드로 변환
    img = tf.convert_to_tensor(np.array(img))  # 텐서로 변환
    return img

# 객체 탐지 실행 함수
def run_detector(detector, img_path):
    img = load_img(img_path)
    converted_img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    
    start_time = time.time()
    result = detector(converted_img)
    end_time = time.time()
    
    result = {key: value.numpy() for key, value in result.items()}
    
    print(f"Found {len(result['detection_scores'])} objects.")
    print(f"Inference time: {end_time - start_time:.2f} seconds")
    
    return img, result

def draw_boxes(image, result):
    draw = ImageDraw.Draw(image)
    
    # 이미지 크기에 따라 폰트 크기와 선 두께 조정
    font_size = int(image.width / 20)  # 이미지 가로 크기의 1/20을 폰트 크기로 설정
    line_width = int(image.width / 100)  # 이미지 가로 크기의 1/100을 선 두께로 설정
    
    font = ImageFont.truetype('arial.ttf', size=font_size)
    
    scores = result['detection_scores']
    boxes = result['detection_boxes']
    classes = result['detection_class_entities']
    
    if np.isscalar(scores):
        print("Warning: scores is a scalar value. No boxes will be drawn.")
        return image
    
    for score, box, class_entity in zip(scores, boxes, classes):
        if score > 0.5:
            ymin, xmin, ymax, xmax = box
            (left, right, top, bottom) = (xmin * image.width, xmax * image.width,
                                          ymin * image.height, ymax * image.height)
            draw.rectangle((left, top, right, bottom), outline='red', width=line_width)
            label = f"{class_entity.decode('utf-8')}: {score:.2f}"
            draw.text((left, top), label, font=font, fill='red')
    
    return image

def prc(imgPath):
    # GPU 설정
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU is available and configured")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU found. Using CPU.")

    # 객체 탐지 모델 로드
    module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
    detector = hub.load(module_handle).signatures['default']
    try:
        # 메인 실행 코드
        image_path = imgPath
        img, result = run_detector(detector, image_path)

        # 결과 시각화
        img_with_boxes = draw_boxes(Image.fromarray(img.numpy()), result)

        # 결과 이미지 저장
        img_with_boxes.save(imgPath)
        print(f"결과 이미지가 {imgPath}에 저장되었습니다.")
    except Exception as e:
        print(f"오류가 발생했습니다: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        imgPath = os.path.join('temp', "n.jpg")
        prc(imgPath=imgPath)
    except Exception as e:
        print(f"프로그램 실행 중 오류가 발생했습니다: {str(e)}")
        import traceback
        traceback.print_exc()