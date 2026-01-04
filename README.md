# Ergonocam — Documentación detallada

Este documento describe en detalle las partes importantes del proyecto, el uso de `small640.pt`, la arquitectura del código y los requisitos de software y hardware recomendados.

**Resumen del proyecto:**
- Proyecto que combina detección de postura basada en un modelo YOLOv5 y MediaPipe con dos interfaces principales:
  - Juego interactivo: `app.py` (Pygame) — experiencia visual y audio llamada "Sky Walker".

**Archivos principales (ubicación relativa):**
- [app.py](app.py): Juego y UI principal (Pygame) que integra la cámara, detección en background y lógica de juego.
- [posture_detector.py](posture_detector.py): Clase `PostureDetector` reutilizable que encapsula la inferencia YOLO + MediaPipe y devuelve imagen+datos.
- [check_classes.py](check_classes.py): Script ligero para cargar `small640.pt` y listar las clases del modelo.
- [requirements_windows.txt](requirements_windows.txt): Lista de dependencias sugeridas para Windows.
- [small640.pt](small640.pt): Archivo de pesos del modelo YOLOv5 usado por el proyecto.
- Archivos de audio: `goku.mp3`, `homero.mp3`, `messi.mp3`, `cristiano.mp3` — usados para feedback sonoro.

**Descripción de `small640.pt`**
- `small640.pt` es el archivo de pesos (modelo) entrenado/convertido para YOLOv5 usado en este proyecto. Se utiliza para detectar una clase específica (probablemente postura mala / una etiqueta del dataset). El archivo se carga mediante `yolov5.load(MODEL_PATH)`.
- En el código: `check_classes.py` carga el modelo y lista `model.names` — esto permite verificar las clases incluidas.
- Consideraciones de compatibilidad: las versiones de PyTorch y YOLOv5 deben ser compatibles. En `posture_detector.py` hay un parche temporal sobre `torch.load` para mitigar incompatibilidades entre versiones nuevas de PyTorch y carga de modelos legacy.

**Explicación detallada de bloques relevantes en `app.py`**
- `VisionThread` (hilo de visión):
  - Ubicación: clase `VisionThread` al inicio de `app.py`.
  - Qué hace: abre la cámara (`cv2.VideoCapture(0)`), carga el modelo YOLO (`yolov5.load(MODEL_PATH)`) y procesa frames en bucle en un hilo daemon.
  - Salidas/estado expuesto: `self.latest_frame` (frame para UI), `self.back_bad` (detecta mala postura de espalda por clase YOLO), `self.leg_bad` (ángulo de pierna fuera de rango), `self.leg_angle` (valor numérico de ángulo), `self.hand_raised` (gesto detectado por MediaPipe), `self.yolo_box`, `self.pose_points`.
  - Detección YOLO: redimensiona el frame, ejecuta `self.model(small_frame)` y usa `results.xyxy[0]` para extraer la primera detección.
  - MediaPipe: usa `mp.solutions.pose.Pose` para obtener landmarks y calcula ángulo entre cadera-rodilla-tobillo con `calcular_angulo()`.

- Lógica de ángulos y postura:
  - Función: `calcular_angulo(a, b, c)` calcula el ángulo ABC en grados usando producto punto y acos.
  - Uso: en `VisionThread` y `PostureDetector` para determinar si la rodilla está en el rango esperado (80°–100°). Si no, se marca `leg_bad` / `leg_status = "bad"`.

- Interfaz y render (UI) en `app.py`:
  - Parte de interfaz: dentro de la clase `SmoothGame`.
  - Composición: pantalla se divide en la sección de cámara (izquierda) y área de juego (derecha). Se usan `pygame` surfaces para renderizar el juego, la cámara y la barra inferior (`bottom_bar_surf`).
  - Controles: mouse y teclado procesados en `run()` (ej. `K_SPACE` para iniciar, `K_q` para salir). También hay interacción por gestos (`hand_raised`) para comenzar/reiniciar.

- Audio en el proyecto:
  - Archivos: `goku.mp3`, `homero.mp3`, `messi.mp3`, `cristiano.mp3`.
  - En `SmoothGame`: `pygame.mixer.init()` y `self.loaded_sounds` cargan varios mp3; cuando se detecta mala postura y pasa el cooldown (4s), se reproduce un audio aleatorio para dar feedback auditivo.

- Lógica de juego relevante:
  - Estados: `STATE_MENU`, `STATE_PLAYING`, `STATE_STRUGGLING`, `STATE_GAMEOVER`.
  - Variables críticas: `instability` (inestabilidad acumulada), `balance` (inclinación), `bad_posture_timer` (tiempo acumulado en mala postura), `good_posture_timer`, `final_fall_direction`.
  - Transiciones: mala postura por más de 3s puede mover el juego a `STATE_STRUGGLING`; si `instability >= 100` se produce `STATE_GAMEOVER` y se activa el ragdoll físico.

- Física y efectos:
  - `ProceduralStickman`: calcula posiciones de articulaciones de forma procedural y dibuja el stickman.
  - `PhysicalRagdoll`: utiliza `pymunk` para crear cuerpos y restricciones; se activa al caer para simular física realista.

**Descripción de `posture_detector.py`**
- Clase `PostureDetector` encapsula la inferencia YOLO + MediaPipe y ofrece una API simple:
  - `process_frame()` → retorna `(PIL.Image, result_data)` donde `result_data` incluye `status` (`good`/`bad`), `yolo_status`, `leg_status` y `angle`.
  - Incluye un parche (monkey-patch) para `torch.load` para forzar `weights_only=False` en entornos donde `yolov5.load` podría fallar con versiones nuevas de PyTorch.
  - Dibuja un overlay sencillo en el frame convertido a RGB para mostrar la línea de la pierna y un marcador de la rodilla.

**Descripción de `check_classes.py`**
- Script pequeño que carga `small640.pt` con `yolov5.load()` y lista `model.names` (IDs y nombres de clase). Sirve para validar que el modelo se carga en el entorno.

**Requerimientos de software (propuestos)**
- Lenguaje: Python 3.10+ (proyecto usa `venv310` en el repositorio).
- Librerías principales (ver [requirements_windows.txt](requirements_windows.txt)):
  - `torch`, `torchvision`, `torchaudio` (versiones compatibles con CUDA/CPU)
  - `yolov5` (interfaz de carga de modelos YOLOv5)
  - `opencv-python`, `mediapipe`, `numpy`
  - `pygame`, `pymunk` (física), `customtkinter`, `Pillow`, `playsound`
- Sistema operativo soportado: Windows 10/11 (soporte probado). El código es portable pero puede requerir ajustes en macOS/Linux para dependencias de `pygame`, `mediapipe` y drivers de cámara.

**Hardware mínimo (sugerido)**
- CPU: Intel i5 / AMD Ryzen 5 (4+ cores)
- RAM: 8 GB
- GPU: opcional (sin GPU funcionará en CPU, pero la inferencia será más lenta)
- Cámara: webcam USB o integrada (índice 0)

**Hardware recomendado**
- CPU: Intel i7 / AMD Ryzen 7
- RAM: 16 GB+
- GPU: NVIDIA con soporte CUDA (por ejemplo GTX 1660 / RTX 2060 o superior) para acelerar PyTorch/YOLOv5

**FPS esperados y consideraciones de rendimiento**
- Objetivo: 20–30 FPS en detección en sistemas con GPU; 5–12 FPS en CPU dependiendo del modelo y tamaño del frame.
- El código en `app.py` reduce la carga: hace inferencia YOLO cada 3 frames y redimensiona a (320x240) para acelerar.

**Sistemas Operativos compatibles**
- Primario: Windows 10/11 (provisto `requirements_windows.txt`).
- Secundario: Linux (Ubuntu 20.04+) y macOS — se pueden necesitar cambios en paquetes, build de mediapipe o versiones alternativas de `torch`.

**Cómo ejecutar (resumen)**
1. Crear y activar entorno virtual:

```powershell
python -m venv venv310
venv310\\Scripts\\activate
```

2. Instalar dependencias:

```powershell
pip install -r requirements_windows.txt
```

3. Probar carga del modelo:

```powershell
python check_classes.py
```

4. Ejecutar el juego Pygame con cámara:

```powershell
python app.py
```

**Consejos y solución de problemas**
- Si `yolov5` no se importa, clona el repo oficial de `ultralytics/yolov5` y añade su carpeta al `PYTHONPATH` o instala una versión empaquetada.
- Si `mediapipe` falla en Windows, instala la versión especificada en `requirements_windows.txt` o usa las ruedas oficiales para tu plataforma.
- Errores de audio: comprueba que los mp3 existen y que `pygame.mixer` o `playsound` pueden acceder al dispositivo de audio.
- Si la cámara no responde, prueba con `cv2.VideoCapture(0)` en un REPL para confirmar índice y permisos.

---
Archivo actualizado: este README amplía detalles técnicos y pasos para reproducir/depurar la aplicación. Si quieres, puedo:
- ejecutar `python check_classes.py` en este entorno para validar `small640.pt`,
- añadir ejemplos de snippets para integrar `PostureDetector` en otros scripts,
- o generar una versión breve del informe en PDF/Markdown listo para presentar.

Generado automáticamente: documentación extendida del proyecto.
