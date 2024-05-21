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

def main():
    global CONFIDENCE_THRESHOLD
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
            
            for track in trackers:
                bbox = track.to_tlwh()  # Obtiene las coordenadas superior izquierda, ancho y alto del bounding box
                bbox = [int(coord) for coord in bbox]  # Convierte las coordenadas a enteros
                det_conf = track.det_conf  # Obtener la confianza del detector
                if det_conf is not None and det_conf > CONFIDENCE_THRESHOLD:
                    area = bbox[2] * bbox[3]
                    if area > largest_area:
                        largest_area = area
                        largest_bbox = bbox
            if largest_bbox is not None:
                # cv2.rectangle(img, (largest_bbox[0], largest_bbox[1]), (largest_bbox[0] + largest_bbox[2], largest_bbox[1] + largest_bbox[3]), (255, 255, 0), 2)
                # Verificar si track.track_id y det_conf son None antes de formatear la cadena
                track_id = track.track_id if track.track_id is not None else "Unknown"
                det_conf_str = f"{det_conf:.2f}" if det_conf is not None else "Unknown"
                text = f"ID: {track_id}, Conf: {det_conf_str}, Área: {largest_area}"
                # cv2.putText(img, text, (largest_bbox[0], largest_bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                center_x = largest_bbox[0] + largest_bbox[2] // 2
                center_y = largest_bbox[1] + largest_bbox[3] // 2
                arduino.write((str(center_x) + ":" + str(center_y) + 'x').encode())
            # cv2.imshow('Object Tracking', img)
            
            #bbox = max_conf_nearest_object(trackers)
            # if bbox is not None:  
            #     aim_absolute(bbox, arduino)
            #     cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            # cv2.imshow('Object Detection', img)
                
            if keyboard.is_pressed('i'):
                classes = [2]
                CONFIDENCE_THRESHOLD = 0.6
                print('ct_head')
            if keyboard.is_pressed('j'):
                classes = [1]
                CONFIDENCE_THRESHOLD = 0.93
                print('ct_body')
            if keyboard.is_pressed('o'):
                classes = [4]
                CONFIDENCE_THRESHOLD = 0.6
                print('t_head')
            if keyboard.is_pressed('k'):
                CONFIDENCE_THRESHOLD = 0.93
                classes = [3]
                print('t_body')
            if keyboard.is_pressed('l'):
                classes = [0]
                print('none')
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Agrega esto para actualizar la ventana de OpenCV
                break
            # if keyboard.is_pressed('q') & 0xFF == ord('q'):
            #     break
    cv2.destroyAllWindows()
        
if __name__ == "__main__":
    main()