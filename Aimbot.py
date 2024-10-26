import mss
from colorama import Fore, Style, init
import numpy as np
import math
import keyboard
import torch 
import serial
from deep_sort_realtime.deepsort_tracker import DeepSort
from PIL import Image
import serial.tools.list_ports
import time

max_age = 5  # Reducir para mantener un seguimiento más corto y fresco de los objetos
max_iou_distance = 0.7  # Ajustar según la superposición requerida entre cuadros delimitadores

deepsort = DeepSort(
    max_age=max_age, 
    max_iou_distance=max_iou_distance,
)

arduino = None
CONFIDENCE_THRESHOLD = None
DETECTION_Y_PORCENT = None
init(autoreset=True)

def print_status(status_message):
    print(f"{Fore.CYAN}{status_message}{Style.RESET_ALL}")
    
def show_instructions():
    print(Fore.CYAN + "Configuración de teclas:" + Style.RESET_ALL)
    print(Fore.YELLOW + "i: ct_head" + Style.RESET_ALL)
    print(Fore.YELLOW + "j: ct_body" + Style.RESET_ALL)
    print(Fore.YELLOW + "o: t_head" + Style.RESET_ALL)
    print(Fore.YELLOW + "k: t_body" + Style.RESET_ALL)
    print(Fore.YELLOW + "l: none" + Style.RESET_ALL)
    print(Fore.RED + "q: salir" + Style.RESET_ALL)
    print(Fore.CYAN + "Presiona una tecla..." + Style.RESET_ALL)
    
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

def aim(bbox, mouse_x, mouse_y, arduino):
    centerX = (bbox[0] + bbox[2] / 2)
    centerY = (bbox[1] + bbox[3] * DETECTION_Y_PORCENT)
    adjusted_center_x, adjusted_center_y = adjust_center(centerX, centerY, mouse_x, mouse_y)
    return arduino.write((str(adjusted_center_x).split('.')[0] + ":" + str(adjusted_center_y).split('.')[0] + 'x').encode())

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
        show_instructions()

        while True:
            if classes != [0]:
                img = np.array(Image.frombytes('RGB', (width, height), sct.grab(monitor).rgb))
                results = model(img)     
                bbs = convert_to_bbs(results, classes)    
                trackers = deepsort.update_tracks(bbs, frame=img)
                largest_bbox = None
                largest_area = 0
                mouse_x, mouse_y = width/2, height/2
                nearest_distance = float('inf')
                max_conf = 0
                for track in trackers:
                    bbox = track.to_tlwh()  # Obtiene las coordenadas superior izquierda, ancho y alto del bounding box
                    bbox = [int(coord) for coord in bbox]  # Convierte las coordenadas a enteros
                    bbox_center_x = int(bbox[0] + bbox[2] / 2)
                    bbox_center_y = int(bbox[1] + bbox[3] * DETECTION_Y_PORCENT)
                    det_conf = track.det_conf  # Obtener la confianza del detector
                    distance = math.sqrt((bbox_center_x - mouse_x)**2 + (bbox_center_y - mouse_y)**2)
                    
                    if det_conf is not None and det_conf > max_conf:
                        area = bbox[2] * bbox[3]
                        if area > largest_area and distance < nearest_distance:
                            max_conf = det_conf
                            nearest_distance = distance
                            largest_area = area
                            largest_bbox = bbox
                if largest_bbox is not None:
                    aim(largest_bbox, mouse_x, mouse_y, arduino)
                    time.sleep(0.02)
            if keyboard.is_pressed('i'):
                classes = [2]
                CONFIDENCE_THRESHOLD = 0.8
                DETECTION_Y_PORCENT = 0.5
                print_status(f"[INFO] Configuración: ct_head, confidence={CONFIDENCE_THRESHOLD}")
                continue
            if keyboard.is_pressed('j'):
                classes = [1]
                CONFIDENCE_THRESHOLD = 0.9
                DETECTION_Y_PORCENT = 0.1
                print_status(f"[INFO] Configuración: ct_body, confidence={CONFIDENCE_THRESHOLD}")
                continue
            if keyboard.is_pressed('o'):
                classes = [4]
                CONFIDENCE_THRESHOLD = 0.8
                DETECTION_Y_PORCENT = 0.5
                print_status(f"[INFO] Configuración: t_head, confidence={CONFIDENCE_THRESHOLD}")
                continue
            if keyboard.is_pressed('k'):
                classes = [3]
                CONFIDENCE_THRESHOLD = 0.9
                DETECTION_Y_PORCENT = 0.1
                print_status(f"[INFO] Configuración: t_body, confidence={CONFIDENCE_THRESHOLD}")
                continue
            if keyboard.is_pressed('l'):
                classes = [0]
                CONFIDENCE_THRESHOLD = None
                DETECTION_Y_PORCENT = None
                print_status(f"[INFO] Configuración: none")
                continue
            if keyboard.is_pressed('q') & 0xFF == ord('q'):
                break
        
if __name__ == "__main__":
    main()