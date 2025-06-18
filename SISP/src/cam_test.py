from picamera2 import Picamera2, Preview
import cv2
import time

# Inicializa a câmera
picam2 = Picamera2()

# Configura preview de vídeo
video_config = picam2.create_preview_configuration(main={"size": (1920, 1080), "format": "BGR888"})
picam2.configure(video_config)
picam2.start()

# Dá tempo para ajustar exposição e foco
time.sleep(3)

# Ativa auto exposição e balanço de branco
picam2.set_controls({
    "AeEnable": True,
    "AwbEnable": True,
    "AnalogueGain": 1.0,
    "Brightness": 0.3,
    "ExposureTime": 10000
})

print("[INFO] Pressione 'q' para sair.")

# Loop de visualização
while True:
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Corrige as cores

    cv2.imshow("Video da Raspberry Pi Cam", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
