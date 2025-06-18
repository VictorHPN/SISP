import cv2
import numpy as np
import os
import time
import argparse
from datetime import datetime
from picamera2 import Picamera2
import RPi.GPIO as GPIO
import threading
import subprocess
import simpleaudio as sa
import wave

# === Global Configs ===
CONFIDENCE_THRESHOLD = 0.1

# Motor config
MOTOR_FREQ_GO = 1.0
MOTOR_FREQ_STOP = 3.0
MOTOR_FREQ_NONE = 0.2
MOTOR_PWM_DUTY_CYCLE_GO_STOP = 30
MOTOR_PWM_DUTY_CYCLE_NONE = 10
MOTOR_PIN = 19

# Audio play loop time
AUDIO_PLAY_INTERVAL = 2.0

# Button GPIO
BUTTON_PIN = 13

# Dir Paths
TEST_IMGS_DIR = "/home/victor/PTLI/src/imgs"
OUTPUT_DIR = "/home/victor/PTLI/results"
TEST_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "test_mode")
CONFIG_PATH = "/home/victor/PTLI/YOLO/yolov4-tiny-custom-PTLD.cfg"
WEIGHTS_PATH = "/home/victor/PTLI/YOLO/yolov4-tiny-custom-PTLD_best.weights"
CLASSES_PATH = "/home/victor/PTLI/YOLO/obj.names"

# Audio feedback files
AUDIO_GO_PATH = "audio/go.wav"
AUDIO_STOP_PATH = "audio/stop.wav"
AUDIO_SEARCHING_PATH = "audio/searching.wav"
AUDIO_VOLUME = 8.0

# Inference Classes:
CLASS_NONE = 0
CLASS_GO = 1
CLASS_STOP = 2

# === Global Data ===
args = None
classes = []
net = None
ln = None
motor_pwm = None
picam2 = None
camera_mode_active = False
thread_camera = None
stop_camera = threading.Event()
session_active = False
session_dir = None
session_start_time = None
session_inference_count = 0
session_total_inference_time = 0.0
session_go_count = 0
session_stop_count = 0
session_go_conf_sum = 0.0
session_stop_conf_sum = 0.0

# Audio control
audio_thread = None
audio_thread_stop = threading.Event()
last_audio_freq = None
last_audio_time = 0.0

def parse_args():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logs in terminal")
    parser.add_argument("-c", "--camera", action="store_true", help="Use Raspberry Pi camera")
    parser.add_argument("-t", "--test", action="store_true", help="Test mode with images from directory")
    parser.add_argument("-p", "--preview", dest="preview", action="store_true", help="Show real-time preview even without detection")
    args = parser.parse_args()

def initialize_classes():
    global classes
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    with open(CLASSES_PATH, "r") as f:
        classes = f.read().strip().split("\n")

def initialize_model():
    global net, ln
    net = cv2.dnn.readNetFromDarknet(CONFIG_PATH, WEIGHTS_PATH)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    ln = net.getUnconnectedOutLayersNames()

def initialize_gpio():
    global motor_pwm
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(MOTOR_PIN, GPIO.OUT)
    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    motor_pwm = GPIO.PWM(MOTOR_PIN, 1)
    motor_pwm.start(0)

def initialize_camera():
    global picam2
    print("[INFO] Initializing camera...")
    picam2 = Picamera2()
    video_config = picam2.create_preview_configuration(main={"size": (1920, 1080), "format": "BGR888"})
    picam2.configure(video_config)
    picam2.start()
    time.sleep(3)
    picam2.set_controls({"AeEnable": True, "AwbEnable": True})

def get_cpu_temperature():
    try:
        output = subprocess.check_output(["vcgencmd", "measure_temp"]).decode()
        temp = float(output.replace("temp=", "").replace("'C\n", ""))
        return temp
    except Exception:
        return -1  # Could not get temperature

def start_session():
    global session_active, session_dir, session_start_time
    global session_inference_count, session_total_inference_time
    global session_go_count, session_stop_count
    global session_go_conf_sum, session_stop_conf_sum
    global args

    session_start_time = datetime.now()
    if args.test:
        session_dir = TEST_OUTPUT_DIR
    else:
        session_dir = os.path.join(OUTPUT_DIR, session_start_time.strftime("session_%Y%m%d_%H%M%S"))
        os.makedirs(session_dir, exist_ok=True)
    session_active = True

    # Zera estatísticas da sessão
    session_inference_count = 0
    session_total_inference_time = 0.0
    session_go_count = 0
    session_stop_count = 0
    session_go_conf_sum = 0.0
    session_stop_conf_sum = 0.0

    print(f"[SESSION] Started new session at {session_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[SESSION] Saving images to: {session_dir}")

def end_session():
    global session_active

    if not session_active:
        return

    session_end_time = datetime.now()
    duration = (session_end_time - session_start_time).total_seconds()
    avg_inference_time = session_total_inference_time / session_inference_count if session_inference_count > 0 else 0
    avg_go_conf = session_go_conf_sum / session_go_count if session_go_count > 0 else 0
    avg_stop_conf = session_stop_conf_sum / session_stop_count if session_stop_count > 0 else 0
    temperature = get_cpu_temperature()

    report_path = os.path.join(session_dir, "session_report.txt")
    with open(report_path, "w") as f:
        f.write("Session Report\n")
        f.write(f"Start Time: {session_start_time}\n")
        f.write(f"End Time: {session_end_time}\n")
        f.write(f"Duration (s): {duration:.2f}\n")
        f.write(f"Inference Count: {session_inference_count}\n")
        f.write(f"Average Inference Time (s): {avg_inference_time:.2f}\n")
        f.write(f"CPU Temperature (°C): {temperature}\n\n")
        f.write(f"Go Detections: {session_go_count}\n")
        f.write(f"Average Go Confidence: {avg_go_conf:.2f}\n")
        f.write(f"Stop Detections: {session_stop_count}\n")
        f.write(f"Average Stop Confidence: {avg_stop_conf:.2f}\n")

    stop_motor_and_audio()

    print(f"[SESSION] Session ended. Report saved to: {report_path}")
    session_active = False

def save_detection_result(image, detected_class=CLASS_NONE, filename=None, test_mode=False):
    # Checks the output path to save the image
    if test_mode:
        output_path = TEST_OUTPUT_DIR
    elif session_active:
        output_path = session_dir
    else:
        output_path = OUTPUT_DIR
    os.makedirs(output_path, exist_ok=True)

    # Checks the inferred object class to add to the image file name 
    result_class = ""
    if detected_class == CLASS_GO:
        result_class = "go"
    elif detected_class == CLASS_STOP:
        result_class = "stop"
    else:
        result_class = "none"

    # If no file name was specified, the default file name is used (img_<timestamp>) 
    if filename == None:
        filename = datetime.now().strftime("img_output_%Y%m%d_%H%M%S")

    # Save the received image
    save_path = os.path.join(output_path, f"{filename}_{result_class}.jpg")
    cv2.imwrite(save_path, image)

    if args.verbose:
        print(f"[INFO] Image saved at: {save_path}")

def set_audio_feedback(freq):
    global audio_thread, last_audio_freq, last_audio_time

    now = time.time()
    if freq == last_audio_freq and now - last_audio_time < AUDIO_PLAY_INTERVAL:
        return  # evita tocar som repetido ou muito rápido

    last_audio_freq = freq
    last_audio_time = now

    if freq == MOTOR_FREQ_GO:
        audio_path = AUDIO_GO_PATH
    elif freq == MOTOR_FREQ_STOP:
        audio_path = AUDIO_STOP_PATH
    else:
        audio_path = AUDIO_SEARCHING_PATH

    # Interrompe áudio anterior
    if audio_thread and audio_thread.is_alive():
        audio_thread_stop.set()
        audio_thread.join()

    if audio_path:
        audio_thread_stop.clear()
        audio_thread = threading.Thread(target=play_audio_loop, args=(audio_path,), daemon=True)
        audio_thread.start()

def play_audio_loop(audio_path):
    while not audio_thread_stop.is_set():
        try:
            with wave.open(audio_path, 'rb') as wav_file:
                n_channels = wav_file.getnchannels()
                sampwidth = wav_file.getsampwidth()
                framerate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                audio_data = wav_file.readframes(n_frames)

            # Convert audio bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)

            # Apply volume (with clipping to avoid distortion)
            audio_array = np.clip(audio_array * AUDIO_VOLUME, -32768, 32767).astype(np.int16)

            # Create new WaveObject
            wave_obj = sa.WaveObject(audio_array.tobytes(), n_channels, sampwidth, framerate)
            play_obj = wave_obj.play()
            play_obj.wait_done()
        except Exception as e:
            print(f"[AUDIO ERROR] {e}")
        time.sleep(2)  # Wait before playing again

def stop_audio_feedback():
    global audio_thread
    if audio_thread and audio_thread.is_alive():
        audio_thread_stop.set()
        audio_thread.join()

def set_motor_frequency(freq):
    global motor_pwm
    if freq <= 0:
        motor_pwm.ChangeDutyCycle(0)
        if args.verbose:
            print("[INFO] Motor disabled")
    else:
        motor_pwm.ChangeFrequency(freq)

        if freq == MOTOR_FREQ_NONE:
            motor_pwm.ChangeDutyCycle(MOTOR_PWM_DUTY_CYCLE_NONE)
        else:
            motor_pwm.ChangeDutyCycle(MOTOR_PWM_DUTY_CYCLE_GO_STOP)

        if args.verbose:
            print(f"[INFO] Motor freq set to {freq} Hz")

def stop_motor_and_audio():
    # Stop motor
    set_motor_frequency(0)
    # Stop audio thread
    stop_audio_feedback()

def detect_pedestrian_traffic_light(image):
    global session_inference_count, session_total_inference_time
    global session_go_count, session_stop_count
    global session_go_conf_sum, session_stop_conf_sum
    global net

    boxes, confidences, class_ids = [], [], []
    h, w = image.shape[:2]

    # Prepara imagem para a rede
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start_time = time.time()
    outputs = net.forward(ln)
    inference_time = time.time() - start_time

    session_inference_count += 1
    session_total_inference_time += inference_time

    # Coleta as detecções com confiança suficiente
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > CONFIDENCE_THRESHOLD:
                cx, cy, bw, bh = (detection[0:4] * [w, h, w, h]).astype(int)
                x = int(cx - bw / 2)
                y = int(cy - bh / 2)
                boxes.append([x, y, bw, bh])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, 0.4)

    detected_class = CLASS_NONE
    freq = MOTOR_FREQ_NONE

    if len(indexes) > 0:
        max_area = 0
        chosen_index = -1

        for i in indexes.flatten():
            x, y, bw, bh = boxes[i]
            area = bw * bh
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"

            # Desenha todas as detecções
            cv2.rectangle(image, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # A detecção com maior área dita a decisão
            if confidences[i] > CONFIDENCE_THRESHOLD and area > max_area:
                max_area = area
                chosen_index = i

        if chosen_index != -1:
            chosen_class = classes[class_ids[chosen_index]].lower()
            chosen_conf = confidences[chosen_index]

            if chosen_class == "go":
                detected_class = CLASS_GO
                freq = MOTOR_FREQ_GO
                session_go_count += 1
                session_go_conf_sum += chosen_conf
            elif chosen_class == "stop":
                detected_class = CLASS_STOP
                freq = MOTOR_FREQ_STOP
                session_stop_count += 1
                session_stop_conf_sum += chosen_conf

    set_motor_frequency(freq)
    set_audio_feedback(freq)

    return detected_class

def test_mode():
    image_files = [f for f in os.listdir(TEST_IMGS_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    if not image_files:
        print(f"[ERROR] No images found at {0}." .format(TEST_IMGS_DIR))
        return
    
    start_session()

    for img_name in image_files:
        img_path = os.path.join(TEST_IMGS_DIR, img_name)
        image = cv2.imread(img_path)
        if image is None:
            print(f"[WARNING] Could not load {img_name}. Skipping...")
            continue

        inference_result_class = detect_pedestrian_traffic_light(image)

        save_detection_result(image, detected_class=inference_result_class, filename=img_name, test_mode=True)

    end_session()
    print("[END] Image processing finished.")
    GPIO.cleanup()

def camera_infer_loop():
    global picam2
    if args.verbose:
        print("[INFO] Inference loop has started.")
    try:
        while not stop_camera.is_set():
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            inference_result_class = detect_pedestrian_traffic_light(frame)

            save_detection_result(frame, detected_class=inference_result_class)

            if args.preview:
                try:
                    cv2.imshow("Raspberry Pi Cam video", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("[INFO] `q` key pressed. Finishing preview.")
                        stop_camera.set()
                        break
                except Exception as error:
                    print(f"[ERROR] Preview mode error, could not show image: {error}")

    except Exception as error:
        print(f"[ERROR] ERROR at inference loop: {error}")

    finally:
        stop_motor_and_audio()
        if args.verbose:
            print("[INFO] Inference loop was finished.")

def toggle_camera(channel):
    global camera_mode_active, thread_camera
    if camera_mode_active:
        print("[BUTTON] Stopping camera inference.")
        stop_camera.set()
        if thread_camera and thread_camera.is_alive():
            thread_camera.join()
        end_session()
    else:
        print("[BUTTON] Starting camera inference.")
        stop_camera.clear()
        start_session()
        thread_camera = threading.Thread(target=camera_infer_loop, daemon=True)
        thread_camera.start()
    camera_mode_active = not camera_mode_active

def main():
    global thread_camera
    try:
        parse_args()
        initialize_gpio()
        initialize_classes()
        initialize_model()

        if args.test:
            test_mode()
        elif args.camera:
            initialize_camera()

            if args.preview:
                print("[INFO] Preview mode enabled. Button is disabled.")
                stop_camera.clear()
                thread_camera = threading.Thread(target=camera_infer_loop, daemon=True)
                thread_camera.start()
            else:
                GPIO.add_event_detect(BUTTON_PIN, GPIO.RISING, callback=toggle_camera, bouncetime=300)
                print("[INFO] Camera initialized. Use the button to start/stop inference.")

            while True:
                time.sleep(1)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
        stop_camera.set()
        if thread_camera and thread_camera.is_alive():
            thread_camera.join()

    finally:
        if picam2:
            picam2.close()
        if motor_pwm:
            motor_pwm.stop()
        GPIO.cleanup()
        print("[INFO] System shut down.")

if __name__ == "__main__":
    main()
