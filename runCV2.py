import subprocess
import os

def activate_venv():
    activate_path = os.path.join('myenv', 'Scripts', 'activate')
    subprocess.run([activate_path], shell=True)
    
def deactivate_venv():
    deactivate_path = os.path.join('myenv', 'Scripts', 'deactivate')
    subprocess.run([deactivate_path], shell=True)

def run_aimbot():
    try:
        subprocess.run(['python', 'AimbotCV2.py'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error al ejecutar el script AimbotCV2.py: {e}")
        # Desactivar entorno virtual
        deactivate_venv()

if __name__ == "__main__":
    # Activar entorno virtual
    activate_venv()
    run_aimbot()