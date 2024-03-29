import subprocess
import os

def install_git():
    # Verificar si Git está instalado
    try:
        subprocess.run(['git', '--version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print("Git already installed.")
    except subprocess.CalledProcessError:
        # Si Git no está instalado, instalarlo
        print("Installing Git...")
        subprocess.run(['winget', 'install', '--id', 'Git.Git', '-e', '--source', 'winget'], check=True)

def install_dependencies():
    # Cambiar al directorio yolov5
    os.chdir('yolov5')
    # Instalar las dependencias desde el archivo requirements.txt
    print("Installing dependencies...")
    subprocess.run(['pip', 'install', '-r', 'requirements.txt'], check=True)

def clone_repository():
    # Verificar si el repositorio ya está clonado
    if os.path.exists('yolov5'):
        print("YOLOv5 repository already cloned.")
    else:
        # Clonar el repositorio de YOLOv5
        print("Cloning YOLOv5 repository...")
        subprocess.run(['git', 'clone', 'https://github.com/ultralytics/yolov5.git'], check=True)
        os.chdir('yolov5')  # Cambiar al directorio clonado

if __name__ == "__main__":
    # Instalar Git fuera del entorno virtual
    install_git()
    
    # Crear y activar un entorno virtual
    subprocess.run(['python', '-m', 'venv', 'myenv'], check=True)
    activate_path = os.path.join('myenv', 'Scripts', 'activate')
    subprocess.run([activate_path], shell=True)
    
    try:
        clone_repository()
        install_dependencies()
    finally:
        deactivate_path = os.path.join('myenv', 'Scripts', 'deactivate')
        subprocess.run([deactivate_path], shell=True)