import os
import time
import threading
import argparse
from datetime import datetime
from picamera2 import Picamera2
import cv2
import RPi.GPIO as GPIO

# === Configurações ===
BUTTON_PIN = 13
MOTOR_PIN = 19
IMAGE_INTERVAL = 1.0  # segundos entre capturas
OUTPUT_DIR = "/home/victor/PTLI/streets_photos"
IMG_WIDTH = 1920
IMG_HEIGHT = 1080

# PWM do motor
MOTOR_PWM = None

# === Variáveis Globais ===
args = None
picam2 = None
stop_event = threading.Event()
session_prefix = None
session_active = False
session_counter = 0
image_counter = 0

def parse_args():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", help="Logs detalhados no terminal")
    args = parser.parse_args()

def initialize_camera():
    global picam2
    print("[INFO] Inicializando câmera...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (IMG_WIDTH, IMG_HEIGHT), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    time.sleep(3)
    picam2.set_controls({"AeEnable": True, "AwbEnable": True})
    print("[INFO] Câmera pronta.")

def initialize_gpio():
    global MOTOR_PWM
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    GPIO.setup(MOTOR_PIN, GPIO.OUT)
    MOTOR_PWM = GPIO.PWM(MOTOR_PIN, 1)  # inicializa com 1Hz
    MOTOR_PWM.start(0)  # duty cycle 0%
    GPIO.add_event_detect(BUTTON_PIN, GPIO.RISING, callback=handle_button, bouncetime=500)

def update_motor_pwm(freq, duty):
    MOTOR_PWM.ChangeFrequency(freq)
    MOTOR_PWM.ChangeDutyCycle(duty)
    if args.verbose:
        print(f"[MOTOR] Frequência: {freq} Hz | Duty Cycle: {duty}%")

def start_session():
    global session_prefix, session_active, image_counter, session_counter
    session_counter += 1
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_prefix = f"session_{session_counter}_{timestamp}"
    image_counter = 1
    session_active = True
    update_motor_pwm(freq=1.0, duty=30.0)
    print(f"[SESSION] Iniciando nova sessão: {session_prefix}")

def stop_session():
    global session_active
    session_active = False
    update_motor_pwm(freq=0.1, duty=10.0)
    print("[SESSION] Sessão finalizada.")

def capture_images():
    global session_active, image_counter
    while not stop_event.is_set():
        if session_active:
            frame = picam2.capture_array()
            filename = f"{session_prefix}_img_{image_counter:03d}.jpg"
            filepath = os.path.join(OUTPUT_DIR, filename)
            cv2.imwrite(filepath, frame)
            if args.verbose:
                print(f"[IMG] Foto salva em {filepath}")
            image_counter += 1
        time.sleep(IMAGE_INTERVAL)

def handle_button(channel):
    global session_active
    if session_active:
        stop_session()
        stop_event.set()
    else:
        start_session()
        stop_event.clear()
        threading.Thread(target=capture_images, daemon=True).start()

def main():
    parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    initialize_gpio()
    initialize_camera()

    update_motor_pwm(freq=0.1, duty=10.0)  # motor em repouso
    print("[INFO] Aguardando botão para iniciar/parar sessão...")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[INFO] Encerrando programa...")
    finally:
        stop_event.set()
        MOTOR_PWM.stop()
        picam2.stop()
        GPIO.cleanup()
        print("[INFO] GPIO, câmera e motor finalizados.")

if __name__ == "__main__":
    main()
