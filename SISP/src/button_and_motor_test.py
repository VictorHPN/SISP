import RPi.GPIO as GPIO
import time

# === Configurações ===
BUTTON_PIN = 13      # Pino do botão
MOTOR_PIN = 19       # Pino do motor (PWM por hardware)

FREQ_GO = 1.0        # Frequência para "Go"
FREQ_STOP = 2.0      # Frequência para "Stop"

# === Setup GPIO ===
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(MOTOR_PIN, GPIO.OUT)

pwm = GPIO.PWM(MOTOR_PIN, FREQ_GO)
pwm.start(0)  # Começa com motor desligado

modo_go = True

def alternar_estado(channel):
    global modo_go
    modo_go = not modo_go
    if modo_go:
        print("[GO] Frequência 1Hz")
        pwm.ChangeFrequency(FREQ_GO)
    else:
        print("[STOP] Frequência 2Hz")
        pwm.ChangeFrequency(FREQ_STOP)
    pwm.ChangeDutyCycle(50)

# Adiciona detecção de borda de subida no botão
GPIO.add_event_detect(BUTTON_PIN, GPIO.RISING, callback=alternar_estado, bouncetime=300)

print("[INFO] Teste iniciado. Pressione o botão para alternar. Ctrl+C para sair.")

try:
    while True:
        time.sleep(1)  # Mantém o script vivo

except KeyboardInterrupt:
    print("\n[INFO] Encerrando teste...")

finally:
    pwm.stop()
    GPIO.cleanup()
