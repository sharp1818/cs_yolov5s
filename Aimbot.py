import mss
import numpy as np
# import cv2
import keyboard
import torch 
import serial
import pyautogui
from deep_sort_realtime.deepsort_tracker import DeepSort
from PIL import Image
import serial.tools.list_ports

max_age = 5  # Reducir para mantener un seguimiento más corto y fresco de los objetos
max_iou_distance = 0.7  # Ajustar según la superposición requerida entre cuadros delimitadores

deepsort = DeepSort(
    max_age=max_age, 
    max_iou_distance=max_iou_distance,
)

arduino = None
CONFIDENCE_THRESHOLD = 0.8
DETECTION_Y_PORCENT = 0.8
# COLORS = [(100,100,100),(0, 255, 0), (150, 150, 0), (0, 0, 255), (255, 0, 0)]

def detect_arduino_port():
    arduino_ports = []
    for port in serial.tools.list_ports.comports():
        if 'Arduino' in port.manufacturer:
            arduino_ports.append(port.device)
    return arduino_ports

def init_arduino():
    arduino_ports = detect_arduino_port()
    if arduino_ports:
        port = arduino_ports[0]
        return serial.Serial(port, 115200, timeout=0)
    else:
        print("No se detectaron puertos de Arduino. Verifica la conexión.")
        return(arduino)

def aim(bbox, arduino):
    centerX = int((bbox[2] + bbox[0]) / 2)
    centerX = int((bbox[2] + bbox[0]) / 2)
    centerY = int((bbox[3] + bbox[1]) / 2 - (bbox[3] - bbox[1]) / 2 * DETECTION_Y_PORCENT)
    mouse_x, mouse_y = pyautogui.position()

    moveX = int((centerX - mouse_x))
    moveY = int((-centerY + mouse_y))
    return arduino.write((str(moveX) + ":" + str(moveY) + 'x').encode())

def convert_to_bbs(results, classes):
    bbs = []
    for obj in results.xyxy[0].tolist():
        bbox = [float(obj[0]), float(obj[1]), float(obj[2] - obj[0]), float(obj[3] - obj[1])]
        confidence = float(obj[4])
        class_id = int(obj[5])
        if confidence > CONFIDENCE_THRESHOLD and class_id in classes:
            bbs.append((bbox, confidence, class_id))
    return bbs

def main():
    arduino = init_arduino()
    if not arduino:
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load('./yolov5', 'custom', 'train/best.pt', source='local').to(device)

    with mss.mss() as sct: 
        monitor_number = 1
        mon = sct.monitors[monitor_number]
        width = 1920
        height = 1080
        monitor = {
            "top": mon["top"],
            "left": mon["left"],
            "width": width,
            "height": height,
            "mon": monitor_number,
        }
        
        classes = [0]
    
        while True:
            img = np.array(Image.frombytes('RGB', (width, height), sct.grab(monitor).rgb))
            results = model(img)
            
            bbs = convert_to_bbs(results, classes)    
            max_det_conf = 0
            max_det_conf_bbox = None
            # max_det_conf_class = None

            trackers = deepsort.update_tracks(bbs, frame=img)

            for track in trackers:
                if track.is_confirmed():
                    bbox = track.to_ltrb()
                    # det_class = track.det_class
                    det_conf = track.det_conf
                    if det_conf is not None and det_conf > max_det_conf:
                        max_det_conf = det_conf
                        max_det_conf_bbox = bbox
                        # max_det_conf_class = det_class
                        
            if max_det_conf_bbox is not None:
                aim(max_det_conf_bbox, arduino)
            #     cv2.rectangle(img, (int(max_det_conf_bbox[0]), int(max_det_conf_bbox[1])), (int(max_det_conf_bbox[2]), int(max_det_conf_bbox[3])), COLORS[max_det_conf_class], 2)
            # cv2.imshow("Object Detection", img)
        
            if keyboard.is_pressed('j'):
                classes = [1, 2]
                print('ct')
            if keyboard.is_pressed('k'):
                classes = [3, 4]
                print('t')
            if keyboard.is_pressed('o'):
                classes = [0]
                print('none')
            # if keyboard.is_pressed('q') or cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            if keyboard.is_pressed('q') & 0xFF == ord('q'):
                break

        # cv2.destroyAllWindows()
        
if __name__ == "__main__":
    main()