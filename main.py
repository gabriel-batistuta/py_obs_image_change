import cv2
import mediapipe as mp
import numpy as np
import pyvirtualcam
from pyvirtualcam import PixelFormat
import os
import time
from glob import glob

# Carregar módulo v4l2loopback
os.system('''sudo modprobe -r v4l2loopback && sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="FakeCam" exclusive_caps=1''')

# Inicializar rastreamento facial e segmentação
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Configurar captura da webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro ao acessar a webcam!")
    exit()

# Configurar webcam virtual
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20

print(f"Resolução da webcam: {width}x{height} @ {fps} FPS")

# Carregar imagens de avatar e fundos
avatar_folder = "assets/faces"  # Pasta com avatares
background_folder = "assets/backgrounds"  # Pasta com fundos

avatar_files = glob(f"{avatar_folder}/*.png") + glob(f"{avatar_folder}/*.jpg")
background_files = glob(f"{background_folder}/*.png") + glob(f"{background_folder}/*.jpg")

if not avatar_files:
    print("Nenhum avatar encontrado na pasta especificada!")
    exit()

if not background_files:
    print("Nenhum fundo encontrado na pasta especificada!")
    exit()

# Inicializar índices e carregar imagens iniciais
current_avatar_index = 0
current_background_index = 0

avatar = cv2.imread(avatar_files[current_avatar_index], cv2.IMREAD_UNCHANGED)
background = cv2.imread(background_files[current_background_index])
background = cv2.resize(background, (width, height))

last_avatar_change = time.time()
last_background_change = time.time()

# Webcam virtual
with pyvirtualcam.Camera(width, height, fps, fmt=PixelFormat.BGR) as cam:
    print(f"Webcam virtual iniciada: {cam.device}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar frame!")
            break

        # Alterar avatar a cada 5 minutos
        if time.time() - last_avatar_change > 5 * 60:  # 5 minutos
            current_avatar_index = (current_avatar_index + 1) % len(avatar_files)
            avatar = cv2.imread(avatar_files[current_avatar_index], cv2.IMREAD_UNCHANGED)
            last_avatar_change = time.time()
            print(f"Mudando para o avatar: {avatar_files[current_avatar_index]}")

        # Alterar fundo a cada 3 minutos
        if time.time() - last_background_change > 3 * 60:  # 3 minutos
            current_background_index = (current_background_index + 1) % len(background_files)
            background = cv2.imread(background_files[current_background_index])
            background = cv2.resize(background, (width, height))
            last_background_change = time.time()
            print(f"Mudando para o fundo: {background_files[current_background_index]}")

        # Converter imagem para RGB (requerido pelo Mediapipe)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Segmentar fundo
        segmentation_results = selfie_segmentation.process(frame_rgb)

        # Criar máscara binária do plano de fundo
        mask = segmentation_results.segmentation_mask > 0.5
        mask = np.stack((mask,) * 3, axis=-1)  # Tornar a máscara compatível com o frame

        # Aplicar o novo fundo
        frame_with_background = np.where(mask, frame, background)

        # Detectar rosto
        results = face_detection.process(frame_rgb)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, c = frame.shape

                # Calcular a posição e o tamanho do avatar baseado no rosto completo
                x1 = int(bboxC.xmin * w)
                y1 = int(bboxC.ymin * h)
                x2 = int((bboxC.xmin + bboxC.width) * w)
                y2 = int((bboxC.ymin + bboxC.height) * h)

                # Ajustar dimensões para evitar sair do frame
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # Redimensionar o avatar para cobrir o rosto inteiro
                avatar_resized = cv2.resize(avatar, (x2 - x1, y2 - y1))

                # Sobrepor o avatar no local detectado
                for i in range(y2 - y1):
                    for j in range(x2 - x1):
                        if avatar_resized[i, j, 3] > 0:  # Verificar canal alfa
                            frame_with_background[y1 + i, x1 + j] = avatar_resized[i, j, :3]

        # Enviar o frame final para a webcam virtual
        cam.send(frame_with_background)
        cam.sleep_until_next_frame()

        # Mostrar a pré-visualização (opcional)
        # cv2.imshow("FakeCam Preview", frame_with_background)

        # Fechar com tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
