import mss
import numpy as np
import math
import keyboard
import torch 
import serial
from deep_sort_realtime.deepsort_tracker import DeepSort
from PIL import Image
import serial.tools.list_ports
import cv2
import time

max_age = 5  # Reducir para mantener un seguimiento más corto y fresco de los objetos
max_iou_distance = 0.7  # Ajustar según la superposición requerida entre cuadros delimitadores

deepsort = DeepSort(
    max_age=max_age, 
    max_iou_distance=max_iou_distance,
)

arduino = None
CONFIDENCE_THRESHOLD = 0.9
DETECTION_Y_PORCENT = 0.5

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

def find_nearest_object(trackers):
    nearest_distance = float('inf')
    nearest_object_bbox = None
    mouse_x, mouse_y = 960, 540
    for track in trackers:
        if track.is_confirmed():
            bbox = track.to_ltrb()
            bbox_center_x = (bbox[0] + bbox[2]) / 2
            bbox_center_y = (bbox[1] + bbox[3]) / 2
            distance = math.sqrt((bbox_center_x - mouse_x)**2 + (bbox_center_y - mouse_y)**2)
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_object_bbox = track.to_ltrb()
    return nearest_object_bbox

def max_conf_nearest_object(trackers):
    max_det_conf = 0
    nearest_distance = float('inf')
    selected_bbox = None
    mouse_x, mouse_y = 960, 540
    for track in trackers:
        if track.is_confirmed():
            det_conf = track.det_conf
            if det_conf is not None:
                bbox = track.to_ltrb()
                bbox_center_x = (bbox[0] + bbox[2]) / 2
                bbox_center_y = (bbox[1] + bbox[3]) / 2
                distance = math.sqrt((bbox_center_x - mouse_x)**2 + (bbox_center_y - mouse_y)**2)
                
                if det_conf > max_det_conf or (det_conf == max_det_conf and distance < nearest_distance):
                    max_det_conf = det_conf
                    nearest_distance = distance
                    selected_bbox = track.to_ltrb()
                
    return selected_bbox

def max_conf_object(trackers):
    max_det_conf = 0
    max_det_conf_bbox = None
    for track in trackers:
        if track.is_confirmed():
            det_conf = track.det_conf
            if det_conf is not None and det_conf > max_det_conf:
                max_det_conf = det_conf
                max_det_conf_bbox = track.to_ltrb()
    if max_det_conf_bbox is not None:
        return max_det_conf_bbox

def firts_object(trackers):
    if trackers:
        first_track = trackers[0]
        if first_track.is_confirmed():
            bbox = first_track.to_ltrb()
            return bbox
    return None

def min_id_object(trackers):
    min_id = float('inf')
    
    for track in trackers:
        if track.track_id < min_id:
            min_id = track.track_id
            min_id_object_bbox = track.to_ltrb()
    if min_id_object_bbox is not None:
        return min_id_object_bbox
    
def max_conf_min_id_object(trackers):
    max_det_conf = 0
    min_id = float('inf')
    max_conf_min_id_bbox = None
    
    for track in trackers:
        if track.det_conf is not None: 
            if track.det_conf > max_det_conf:
                max_det_conf = track.det_conf
                min_id = track.track_id
                max_conf_min_id_bbox = track.to_ltrb()
            elif track.det_conf == max_det_conf:
                if track.track_id < min_id:
                    min_id = track.track_id
                    max_conf_min_id_bbox = track.to_ltrb()
                
    return max_conf_min_id_bbox

def aim(bbox, mouse_x, mouse_y, arduino):
    centerX = int((bbox[2] + bbox[0]) / 2)
    centerY = int((bbox[3] + bbox[1]) / 2 - (bbox[3] - bbox[1]) / 2 * DETECTION_Y_PORCENT)
    moveX = int((centerX - mouse_x))
    moveY = int((-centerY + mouse_y))
  #  print(moveX,moveY)
    return arduino.write((str(moveX) + ":" + str(moveY) + 'x').encode())

def aim_absolute(bbox, arduino):
    centerX = int((bbox[2] + bbox[0]) / 2)
    centerY = int((bbox[3] + bbox[1]) / 2 - (bbox[3] - bbox[1]) / 2 * DETECTION_Y_PORCENT)
    print(centerX, centerY)
    return arduino.write((str(centerX) + ":" + str(centerY) + 'x').encode())

def convert_to_bbs(results, classes):
    bbs = []
    for obj in results.xyxy[0].tolist():
        bbox = [float(obj[0]), float(obj[1]), float(obj[2] - obj[0]), float(obj[3] - obj[1])]
        confidence = float(obj[4])
        class_id = int(obj[5])
        if confidence > CONFIDENCE_THRESHOLD and class_id in classes:
            bbs.append((bbox, confidence, class_id))
    return bbs

def adjust_center(center_x, center_y, mouse_x, mouse_y):
    _center_x = abs(center_x - mouse_x)
    _center_y = abs(center_y - mouse_y)
    adjusted_center_x = center_x
    adjusted_center_y = center_y
    # Ajuste en el eje X (izquierda y derecha)
    if center_x > mouse_x:
        if _center_x < mouse_x / 32:
            adjusted_center_x = int(center_x)
        elif _center_x < mouse_x / 16:
            adjusted_center_x = int(center_x * 1.02)
        elif _center_x < mouse_x / 10:
            adjusted_center_x = int(center_x * 1.04)
        elif _center_x < mouse_x / 5:
            adjusted_center_x = int(center_x * 1.06)
        elif _center_x < mouse_x / 3.2:
            adjusted_center_x = int(center_x * 1.08)
        elif _center_x < mouse_x / 2.4:
            adjusted_center_x = int(center_x * 1.1)
        elif _center_x < mouse_x / 1.6:
            adjusted_center_x = int(center_x * 1.09)
        elif _center_x < mouse_x / 1.4:
            adjusted_center_x = int(center_x * 1.074)
        elif _center_x < mouse_x / 1.25:
            adjusted_center_x = int(center_x * 1.056)
        elif _center_x < mouse_x / 1.1:
            adjusted_center_x = int(center_x * 1.036)
        elif _center_x < mouse_x:
            adjusted_center_x = int(center_x * 1.02)
    else:
        if _center_x < mouse_x / 32:
            adjusted_center_x = int(center_x)
        elif _center_x < mouse_x / 16:
            adjusted_center_x = int(center_x * 0.98)
        elif _center_x < mouse_x / 10:
            adjusted_center_x = int(center_x * 0.96)
        elif _center_x < mouse_x / 5:
            adjusted_center_x = int(center_x * 0.94)
        elif _center_x < mouse_x / 3.2:
            adjusted_center_x = int(center_x * 0.88)
        elif _center_x < mouse_x / 2.4:
            adjusted_center_x = int(center_x * 0.8)
        elif _center_x < mouse_x / 1.6:
            adjusted_center_x = int(center_x * 0.72)
        elif _center_x < mouse_x / 1.4:
            adjusted_center_x = int(center_x * 0.65)
        elif _center_x < mouse_x / 1.25:
            adjusted_center_x = int(center_x * 0.58)
        elif _center_x < mouse_x / 1.1:
            adjusted_center_x = int(center_x * 0.5)
        elif _center_x < mouse_x:
            adjusted_center_x = int(center_x * 0.4)
    # Ajuste en el eje Y (arriba y abajo)
    if center_y > mouse_y:
        if _center_y < mouse_y / 6:
            adjusted_center_y = int(center_y * 1.03)
        elif _center_y < mouse_y / 3:
            adjusted_center_y = int(center_y * 1.08)
        elif _center_y < mouse_y * 2 / 3:
            adjusted_center_y = int(center_y * 1.12)
        elif _center_y < mouse_y:
            adjusted_center_y = int(center_y * 1.14)
    else:
        if _center_y < mouse_y / 6:
            adjusted_center_y = int(center_y * 1.03)
        elif _center_y < mouse_y / 3:
            adjusted_center_y = int(center_y * 0.9)
        elif _center_y < mouse_y * 2 / 3:
            adjusted_center_y = int(center_y * 0.7)
        elif _center_y < mouse_y:
            adjusted_center_y = int(center_y * 0.3)

    return adjusted_center_x, adjusted_center_y

def main():
    global CONFIDENCE_THRESHOLD
    global DETECTION_Y_PORCENT
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
        bbox = None
        
        while True:
            img = np.array(Image.frombytes('RGB', (width, height), sct.grab(monitor).rgb))
            results = model(img)     
            bbs = convert_to_bbs(results, classes)    
            trackers = deepsort.update_tracks(bbs, frame=img)
            largest_bbox = None
            largest_area = 0
            mouse_x, mouse_y = 960, 540
            nearest_distance = float('inf')
            for track in trackers:
                bbox = track.to_tlwh()  # Obtiene las coordenadas superior izquierda, ancho y alto del bounding box
                bbox = [int(coord) for coord in bbox]  # Convierte las coordenadas a enteros
                bbox_center_x = (bbox[0] + bbox[2]) / 2
                bbox_center_y = int(bbox[1] + bbox[3] * DETECTION_Y_PORCENT)
                det_conf = track.det_conf  # Obtener la confianza del detector
                distance = math.sqrt((bbox_center_x - mouse_x)**2 + (bbox_center_y - mouse_y)**2)

                # if det_conf is not None and det_conf > CONFIDENCE_THRESHOLD:
                #     cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 255, 0), 2)
                #     # Verificar si track.track_id y det_conf son None antes de formatear la cadena
                #     track_id = track.track_id if track.track_id is not None else "Unknown"
                #     det_conf_str = f"{det_conf:.2f}" if det_conf is not None else "Unknown"
                #     text = f"ID: {track_id}, Conf: {det_conf_str}, Área: {largest_area}"
                #     cv2.putText(img, text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
                try:
                    track_id = int(track.track_id) if track.track_id is not None else float('inf')
                except ValueError:
                    track_id = float('inf')
                
                if det_conf is not None and det_conf > CONFIDENCE_THRESHOLD:
                    area = bbox[2] * bbox[3]
                    # if area > largest_area:
                    #     largest_area = area
                    #     largest_bbox = bbox
                    if area > largest_area and distance < nearest_distance:
                        nearest_distance = distance
                        largest_area = area
                        largest_bbox = bbox
            if largest_bbox is not None:
                # cv2.rectangle(img, (largest_bbox[0], largest_bbox[1]), (largest_bbox[0] + largest_bbox[2], largest_bbox[1] + largest_bbox[3]), (255, 255, 0), 2)
                # Verificar si track.track_id y det_conf son None antes de formatear la cadena
                # track_id = track.track_id if track.track_id is not None else "Unknown"
                # det_conf_str = f"{det_conf:.2f}" if det_conf is not None else "Unknown"
                center_x = largest_bbox[0] + largest_bbox[2] / 2
                center_y = int(largest_bbox[1] + largest_bbox[3] * DETECTION_Y_PORCENT)
                # text = f" Distance: {abs(center_x - mouse_x),abs(center_y - mouse_y)}, ID: {track_id}, Conf: {det_conf_str}, Área: {largest_area}"
                # cv2.putText(img, text, (largest_bbox[0], largest_bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)  
                adjusted_center_x, adjusted_center_y = adjust_center(center_x, center_y, mouse_x, mouse_y)
                arduino.write((str(adjusted_center_x).split('.')[0] + ":" + str(adjusted_center_y).split('.')[0] + 'x').encode())
                time.sleep(0.08)
            # cv2.imshow('Object Tracking', img)
                
            if keyboard.is_pressed('i'):
                classes = [2]
                CONFIDENCE_THRESHOLD = 0.75
                DETECTION_Y_PORCENT = 0.5
                print('ct_head')
                continue
            if keyboard.is_pressed('j'):
                classes = [1]
                CONFIDENCE_THRESHOLD = 0.9
                DETECTION_Y_PORCENT = 0.05
                print('ct_body')
                continue
            if keyboard.is_pressed('o'):
                classes = [4]
                CONFIDENCE_THRESHOLD = 0.75
                DETECTION_Y_PORCENT = 0.5
                print('t_head')
                continue
            if keyboard.is_pressed('k'):
                classes = [3]
                CONFIDENCE_THRESHOLD = 0.9
                DETECTION_Y_PORCENT = 0.05
                print('t_body')
                continue
            if keyboard.is_pressed('l'):
                classes = [0]
                print('none')
                continue
            # if cv2.waitKey(1) & 0xFF == ord('q'):  # Agrega esto para actualizar la ventana de OpenCV
            #     break
            if keyboard.is_pressed('q') & 0xFF == ord('q'):
                break
    # cv2.destroyAllWindows()
        
if __name__ == "__main__":
    main()