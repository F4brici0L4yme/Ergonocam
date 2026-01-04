import cv2
import torch
import yolov5
import mediapipe as mp
import math
import time
import pygame
import numpy as np
import random
import pymunk
import threading
import os 

# --- CONFIGURACIÓN ---
MODEL_PATH = 'small640.pt' 
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 750 
GAME_WIDTH = 600
CAM_WIDTH = 600
BAR_HEIGHT = 50 

# Colores
SKY_TOP = (0, 100, 200)       
SKY_BOTTOM = (180, 220, 255)  
CLOUD_COLOR = (255, 255, 255, 200)
ROPE_BASE = (139, 69, 19)     
ROPE_SHADOW = (90, 45, 10)    
ROPE_HIGHLIGHT = (160, 85, 35) 
STICK_COLOR = (40, 40, 50)
ACCENT_COLOR = (0, 255, 200)
BUTTON_COLOR = (0, 200, 100)
BUTTON_HOVER = (0, 255, 150)
BUTTON_DANGER = (255, 80, 80)
BUTTON_DANGER_HOVER = (255, 120, 120)
BAR_BG_COLOR = (50, 50, 50)
VALID_COLOR = (0, 255, 0)   
INVALID_COLOR = (0, 0, 255) 

# --- HILO DE VISIÓN (IA) ---
class VisionThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.running = True
        
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.latest_frame = None
        self.back_bad = False
        self.leg_bad = False
        self.leg_angle = 0
        self.yolo_box = None
        self.pose_points = None
        self.hand_raised = False

        print('Comprobando CUDA...')
        if torch.cuda.is_available():
            device_idx = torch.cuda.current_device()
            self.device = f'cuda:{device_idx}'
            print(f'Inferencia GPU activada en: {torch.cuda.get_device_name(device_idx)}')
        else:
            self.device = 'cpu'
            print('Inferencia CPU activada')

        try:
            self.model = yolov5.load(MODEL_PATH, device=self.device)
            self.model.conf = 0.50
            self.model.iou = 0.50
            self.model.classes = [0, 1]
            self.model.multi_label = False
            self.model.max_det = 1
        except Exception as e: 
            print(f"Error cargando modelo YOLO: {e}")
            self.model = None

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1, 
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def calcular_angulo(self, a, b, c):
        ba = (a[0]-b[0], a[1]-b[1])
        bc = (c[0]-b[0], c[1]-b[1])
        dot = ba[0]*bc[0] + ba[1]*bc[1]
        mag_ba = math.sqrt(ba[0]**2 + ba[1]**2)
        mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)
        if mag_ba * mag_bc == 0: return 0
        cos_angle = dot / (mag_ba * mag_bc)
        return math.degrees(math.acos(max(min(cos_angle, 1), -1)))

    def run(self):
        frame_count = 0
        orig_w, orig_h = 640, 480
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret: continue
            
            frame_count += 1
            do_inference = (frame_count % 3 == 0)

            if do_inference:
                small_frame = cv2.resize(frame, (320, 240))
                frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                self.back_bad = False 
                self.yolo_box = None
                
                if self.model:
                    try:
                        results = self.model(frame_rgb)
                        df_results = results.pandas().xyxy[0].to_dict(orient="records")
                        
                        if df_results:
                            result = df_results[0]
                            class_id = result['class']  
                            
                            x1 = int(result['xmin']); y1 = int(result['ymin'])
                            x2 = int(result['xmax']); y2 = int(result['ymax'])
                            
                            scale_x = orig_w / 320; scale_y = orig_h / 240
                            self.yolo_box = (int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y))

                            if class_id == 1: 
                                self.back_bad = True
                    except Exception as e:
                        print(f"Error inferencia YOLO: {e}")

                self.leg_bad = False
                self.hand_raised = False
                
                frame_rgb.flags.writeable = False
                mp_results = self.pose.process(frame_rgb)
                
                if mp_results.pose_landmarks:
                    lm = mp_results.pose_landmarks.landmark
                    self.pose_points = {
                        'A': (lm[24].x, lm[24].y), 'B': (lm[26].x, lm[26].y), 'C': (lm[28].x, lm[28].y)
                    }
                    h, w, _ = small_frame.shape
                    pA = (lm[24].x * w, lm[24].y * h)
                    pB = (lm[26].x * w, lm[26].y * h)
                    pC = (lm[28].x * w, lm[28].y * h)
                    self.leg_angle = self.calcular_angulo(pA, pB, pC)
                    
                    if not (80 <= self.leg_angle <= 100):
                        self.leg_bad = True
                        
                    # --- Detección de Mano Levantada ---
                    nose_y = lm[0].y
                    wrist_l_y = lm[15].y
                    wrist_r_y = lm[16].y
                    
                    if wrist_l_y < nose_y or wrist_r_y < nose_y:
                        self.hand_raised = True

            # --- DIBUJO ---
            if self.yolo_box:
                x1, y1, x2, y2 = self.yolo_box
                color = INVALID_COLOR if self.back_bad else VALID_COLOR
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            if self.pose_points:
                pts = self.pose_points
                A = (int(pts['A'][0] * orig_w), int(pts['A'][1] * orig_h))
                B = (int(pts['B'][0] * orig_w), int(pts['B'][1] * orig_h))
                C = (int(pts['C'][0] * orig_w), int(pts['C'][1] * orig_h))
                ln_color = INVALID_COLOR if self.leg_bad else VALID_COLOR
                cv2.line(frame, A, B, ln_color, 3); cv2.line(frame, B, C, ln_color, 3)
                cv2.circle(frame, B, 25, ln_color, 3)
                
                if self.hand_raised:
                    cv2.putText(frame, "MANO LEVANTADA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            self.latest_frame = frame

# --- GESTOR DE ACTIVOS ---
class AssetManager:
    def __init__(self):
        self.clouds = []
        for _ in range(12): self.spawn_cloud(initial=True)
        self.sky_surf = pygame.Surface((GAME_WIDTH, WINDOW_HEIGHT - BAR_HEIGHT))
        self.pre_render_sky()

    def pre_render_sky(self):
        sky_height = WINDOW_HEIGHT - BAR_HEIGHT
        for y in range(sky_height):
            ratio = y / sky_height
            r = SKY_TOP[0] * (1-ratio) + SKY_BOTTOM[0] * ratio
            g = SKY_TOP[1] * (1-ratio) + SKY_BOTTOM[1] * ratio
            b = SKY_TOP[2] * (1-ratio) + SKY_BOTTOM[2] * ratio
            pygame.draw.line(self.sky_surf, (r,g,b), (0, y), (GAME_WIDTH, y))

    def spawn_cloud(self, initial=False):
        x = random.randint(0, GAME_WIDTH) if initial else GAME_WIDTH + 50
        y = random.randint(0, (WINDOW_HEIGHT - BAR_HEIGHT)//2 + 100)
        speed = random.uniform(0.2, 1.5)
        scale = random.uniform(0.6, 1.8)
        blobs = []
        for _ in range(random.randint(4, 8)):
            blobs.append((random.randint(-40, 40), random.randint(-20, 20), random.randint(25, 50)))
        self.clouds.append({'x': x, 'y': y, 'speed': speed, 'scale': scale, 'blobs': blobs})

    def update_clouds(self):
        for c in self.clouds: c['x'] -= c['speed']
        self.clouds = [c for c in self.clouds if c['x'] > -200]
        if len(self.clouds) < 12 and random.random() < 0.01: self.spawn_cloud()

    def draw_clouds(self, surface):
        for c in self.clouds:
            for bx, by, br in c['blobs']:
                cx = int(c['x'] + bx * c['scale'])
                cy = int(c['y'] + by * c['scale'])
                cr = int(br * c['scale'])
                pygame.draw.circle(surface, (200, 210, 230, 80), (cx+10, cy+10), cr)
                pygame.draw.circle(surface, CLOUD_COLOR, (cx, cy), cr)

    def draw_rope_horizontal(self, surface, y_pos, instability):
        vibe = math.sin(time.time() * 50) * (instability / 20.0)
        start_pos = (0 - 10, y_pos + vibe)
        end_pos = (GAME_WIDTH + 10, y_pos + vibe)
        pygame.draw.line(surface, ROPE_SHADOW, (start_pos[0], start_pos[1]+6), (end_pos[0], end_pos[1]+6), 14)
        pygame.draw.line(surface, ROPE_BASE, start_pos, end_pos, 10)
        pygame.draw.line(surface, ROPE_HIGHLIGHT, (start_pos[0], start_pos[1]-3), (end_pos[0], end_pos[1]-3), 4)

# --- STICKMAN ARTICULADO ---
class ProceduralStickman:
    def __init__(self, x, y, scale=1.0):
        self.x = x; self.y = y; self.scale = scale; self.walk_cycle = 0.0
        self.leg_len = 50 * scale; self.torso_len = 45 * scale
        self.arm_len = 60 * scale; self.head_rad = 12 * scale
        self.joints = {}

    def calculate_positions(self, balance_tilt, instability, is_moving, dt, rope_y_pos, struggle_dir):
        if struggle_dir != 0:
            shake_x = random.uniform(-1, 1) * (instability * 0.02)
            shake_y = random.uniform(-1, 1) * (instability * 0.02)
            planted_foot_x = self.x + (5 * struggle_dir) + shake_x
            planted_foot_y = rope_y_pos
            lean_angle = 25 * struggle_dir
            rad_angle = math.radians(lean_angle)
            vx = math.sin(rad_angle); vy = -math.cos(rad_angle)
            hip_x = planted_foot_x + (vx * self.leg_len); hip_y = planted_foot_y + (vy * self.leg_len) + shake_y
            neck_x = hip_x + (vx * self.torso_len); neck_y = hip_y + (vy * self.torso_len)
            lifted_foot_x = self.x - (40 * struggle_dir) + shake_x
            lifted_foot_y = rope_y_pos - 35
            lifted_knee_x = (hip_x + lifted_foot_x) / 2; lifted_knee_y = (hip_y + lifted_foot_y) / 2
            arm_angle_base = rad_angle
            if struggle_dir > 0: l_arm_ang = arm_angle_base - math.radians(85); r_arm_ang = arm_angle_base + math.radians(110)
            else: l_arm_ang = arm_angle_base - math.radians(110); r_arm_ang = arm_angle_base + math.radians(85)
            lhx = neck_x + math.sin(l_arm_ang)*self.arm_len; lhy = neck_y - math.cos(l_arm_ang)*self.arm_len
            rhx = neck_x + math.sin(r_arm_ang)*self.arm_len; rhy = neck_y - math.cos(r_arm_ang)*self.arm_len
            self.joints = {
                'head': (int(neck_x + vx*12), int(neck_y + vy*12)), 'neck': (neck_x, neck_y),
                'hip': (hip_x, hip_y), 'l_knee': (lifted_knee_x, lifted_knee_y), 'l_foot': (lifted_foot_x, lifted_foot_y),
                'r_knee': ((hip_x + planted_foot_x)/2, (hip_y + planted_foot_y)/2), 'r_foot': (planted_foot_x, planted_foot_y),
                'l_hand': (lhx, lhy), 'r_hand': (rhx, rhy)
            }
        else:
            speed = 3.5 if is_moving else 0.5
            self.walk_cycle += dt * speed
            rad_tilt = math.radians(balance_tilt * 0.5)
            hip_x = self.x; hip_y = rope_y_pos - self.leg_len
            neck_x = hip_x + math.sin(rad_tilt) * self.torso_len; neck_y = hip_y - math.cos(rad_tilt) * self.torso_len
            l_ang = math.sin(self.walk_cycle) * 0.4; r_ang = math.sin(self.walk_cycle + math.pi) * 0.4
            l_foot_x = hip_x - 5 + (l_ang * 15); l_foot_y = rope_y_pos - (abs(l_ang) * 5)
            r_foot_x = hip_x + 5 + (r_ang * 15); r_foot_y = rope_y_pos - (abs(r_ang) * 5)
            l_knee_x = (hip_x + l_foot_x)/2 - 5; l_knee_y = (hip_y + l_foot_y)/2
            r_knee_x = (hip_x + r_foot_x)/2 + 5; r_knee_y = (hip_y + r_foot_y)/2
            l_arm_ang = (math.pi/2) + rad_tilt; r_arm_ang = -(math.pi/2) + rad_tilt
            lhx = neck_x + math.sin(l_arm_ang)*self.arm_len; lhy = neck_y + math.cos(l_arm_ang)*self.arm_len
            rhx = neck_x + math.sin(r_arm_ang)*self.arm_len; rhy = neck_y + math.cos(r_arm_ang)*self.arm_len
            self.joints = {
                'head': (int(neck_x + math.sin(rad_tilt)*12), int(neck_y - math.cos(rad_tilt)*12)),
                'neck': (neck_x, neck_y), 'hip': (hip_x, hip_y),
                'l_knee': (l_knee_x, l_knee_y), 'l_foot': (l_foot_x, l_foot_y),
                'r_knee': (r_knee_x, r_knee_y), 'r_foot': (r_foot_x, r_foot_y),
                'l_hand': (lhx, lhy), 'r_hand': (rhx, rhy)
            }

    def draw(self, surface, balance_tilt, instability, is_moving, dt, rope_y_pos, struggle_dir=0):
        self.calculate_positions(balance_tilt, instability, is_moving, dt, rope_y_pos, struggle_dir)
        j = self.joints
        th = 6
        pygame.draw.line(surface, STICK_COLOR, j['hip'], j['l_knee'], th)
        pygame.draw.line(surface, STICK_COLOR, j['l_knee'], j['l_foot'], th)
        pygame.draw.line(surface, STICK_COLOR, j['hip'], j['r_knee'], th)
        pygame.draw.line(surface, STICK_COLOR, j['r_knee'], j['r_foot'], th)
        pygame.draw.line(surface, STICK_COLOR, j['hip'], j['neck'], th+2)
        pygame.draw.line(surface, STICK_COLOR, j['neck'], j['l_hand'], th-1)
        pygame.draw.line(surface, STICK_COLOR, j['neck'], j['r_hand'], th-1)
        pygame.draw.circle(surface, STICK_COLOR, j['head'], int(self.head_rad), 2)
        pygame.draw.circle(surface, (240,230,220), j['head'], int(self.head_rad)-2)

# --- RAGDOLL FÍSICO ---
class PhysicalRagdoll:
    def __init__(self, space, joints):
        self.space = space
        self.bodies = []
        self.shapes = []
        self.constraints = []
        self.body_map = {}

        parts = ['head', 'neck', 'hip', 'l_knee', 'l_foot', 'r_knee', 'r_foot', 'l_hand', 'r_hand']
        for part in parts:
            pos = joints[part]
            mass = 5 if part in ['hip', 'head'] else 2
            radius = 6
            moment = pymunk.moment_for_circle(mass, 0, radius)
            body = pymunk.Body(mass, moment)
            body.position = pos
            shape = pymunk.Circle(body, radius)
            shape.elasticity = 0.1
            shape.friction = 0.5
            shape.filter = pymunk.ShapeFilter(group=1)
            self.space.add(body, shape)
            self.bodies.append(body)
            self.shapes.append(shape)
            self.body_map[part] = body

        connections = [('head', 'neck'), ('neck', 'hip'), ('hip', 'l_knee'), ('l_knee', 'l_foot'), ('hip', 'r_knee'), ('r_knee', 'r_foot'), ('neck', 'l_hand'), ('neck', 'r_hand')]
        for p1, p2 in connections:
            b1 = self.body_map[p1]
            b2 = self.body_map[p2]
            dist = b1.position.get_distance(b2.position)
            joint = pymunk.PinJoint(b1, b2, (0, 0), (0, 0))
            joint.distance = dist
            joint.collide_bodies = False
            self.space.add(joint)
            self.constraints.append(joint)

    def draw(self, surface):
        c = STICK_COLOR; th = 6; b = self.body_map
        pygame.draw.line(surface, c, b['head'].position, b['neck'].position, th)
        pygame.draw.line(surface, c, b['neck'].position, b['hip'].position, th+2)
        pygame.draw.line(surface, c, b['hip'].position, b['l_knee'].position, th)
        pygame.draw.line(surface, c, b['l_knee'].position, b['l_foot'].position, th)
        pygame.draw.line(surface, c, b['hip'].position, b['r_knee'].position, th)
        pygame.draw.line(surface, c, b['r_knee'].position, b['r_foot'].position, th)
        pygame.draw.line(surface, c, b['neck'].position, b['l_hand'].position, th-1)
        pygame.draw.line(surface, c, b['neck'].position, b['r_hand'].position, th-1)
        h_pos = (int(b['head'].position.x), int(b['head'].position.y))
        pygame.draw.circle(surface, c, h_pos, 12, 2)
        pygame.draw.circle(surface, (240,230,220), h_pos, 10)

    def cleanup(self):
        for c in self.constraints: self.space.remove(c)
        for s in self.shapes: self.space.remove(s)
        for b in self.bodies: self.space.remove(b)

# --- JUEGO PRINCIPAL ---
class SmoothGame:
    def __init__(self):
        pygame.init()
        pygame.mixer.init()
        self.audio_files = ["goku.mpeg", "homero.mpeg", "messi.mpeg", "cristiano.mpeg"]
        self.loaded_sounds = []
        try:
            for f in self.audio_files:
                if os.path.exists(f):
                    self.loaded_sounds.append(pygame.mixer.Sound(f))
                else:
                    print(f"Advertencia: No se encontró {f}")
        except Exception as e:
            print(f"Error cargando sonidos: {e}")
        self.last_audio_time = 0
        
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Sky Walker - Audio & Gestures")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Impact", 24)
        self.big_font = pygame.font.SysFont("Impact", 48)

        self.vision = VisionThread()
        self.vision.start()

        self.assets = AssetManager()
        self.space = pymunk.Space()
        self.space.gravity = (0, 900)

        self.rope_y = (WINDOW_HEIGHT - BAR_HEIGHT) // 2 + 100
        self.rope_start = (-100, self.rope_y)
        self.rope_end = (GAME_WIDTH + 100, self.rope_y)

        rope_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        rope_shape = pymunk.Segment(rope_body, self.rope_start, self.rope_end, 5)
        rope_shape.elasticity = 0.5
        rope_shape.friction = 0.8
        rope_shape.filter = pymunk.ShapeFilter(group=1)
        self.space.add(rope_body, rope_shape)

        mid_x = GAME_WIDTH // 2
        self.player = ProceduralStickman(mid_x, self.rope_y, scale=1.5)

        self.STATE_MENU = 0; self.STATE_PLAYING = 1; self.STATE_STRUGGLING = 2; self.STATE_GAMEOVER = 3
        self.current_state = self.STATE_MENU

        self.balance = 0.0; self.instability = 0.0; self.velocity = 0.0
        self.game_over = False
        self.trail = []
        self.ragdoll = None
        self.btn_rect = pygame.Rect(GAME_WIDTH//2 - 100, (WINDOW_HEIGHT - BAR_HEIGHT)//2 - 50, 200, 60)
        self.btn_restart_rect = pygame.Rect(GAME_WIDTH//2 - 100, (WINDOW_HEIGHT - BAR_HEIGHT)//2 + 50, 200, 60)
        self.struggle_start_time = 0; self.final_fall_direction = 0
        
        self.play_start_time = 0
        self.bottom_bar_surf = pygame.Surface((WINDOW_WIDTH, BAR_HEIGHT))
        
        self.bad_posture_timer = 0.0
        self.good_posture_timer = 0.0
        self.last_bad_time = 0.0
        
        self.current_score = 0.0
        self.best_score = 0.0
        
        self.gesture_start_timer = 0.0

    def activate_ragdoll(self):
        self.ragdoll = PhysicalRagdoll(self.space, self.player.joints)
        for name, body in self.ragdoll.body_map.items():
            gx = 300 * self.final_fall_direction
            gy = -100
            rx = random.randint(-1500, 1500); ry = random.randint(-1500, 1500)
            if 'hand' in name or 'foot' in name: rx *= 2; ry *= 2
            body.apply_impulse_at_local_point((gx + rx, gy + ry), (0,0))
            body.angular_velocity = random.uniform(-10, 10)

    def reset_game(self):
        self.balance = 0; self.instability = 0
        self.bad_posture_timer = 0; self.good_posture_timer = 0
        self.last_bad_time = 0
        self.current_score = 0.0
        self.gesture_start_timer = 0.0
        self.current_state = self.STATE_PLAYING
        if self.ragdoll: self.ragdoll.cleanup(); self.ragdoll = None
        self.player.x = GAME_WIDTH // 2
        self.play_start_time = time.time()

    def update(self, dt):
        self.assets.update_clouds()
        
        if self.current_state in [self.STATE_MENU, self.STATE_GAMEOVER]:
            if self.vision.hand_raised:
                self.gesture_start_timer += dt
                if self.gesture_start_timer > 2.0:
                    if self.current_state == self.STATE_MENU:
                        self.current_state = self.STATE_PLAYING
                        self.play_start_time = time.time()
                    else: 
                        self.reset_game()
                    self.gesture_start_timer = 0
            else:
                self.gesture_start_timer = max(0, self.gesture_start_timer - dt)

        if self.current_state == self.STATE_MENU: return
        if self.current_state == self.STATE_GAMEOVER: self.space.step(1/60.0); return

        elapsed_play = time.time() - self.play_start_time
        if elapsed_play < 5.0:
            self.balance = math.sin(elapsed_play * 3) * 5
            self.instability = 0
            self.velocity = 0
            self.bad_posture_timer = 0
            self.current_score = 0.0
            return

        if self.current_state in [self.STATE_PLAYING, self.STATE_STRUGGLING]:
            self.current_score += dt
            if self.current_score > self.best_score:
                self.best_score = self.current_score

            back_bad = self.vision.back_bad
            leg_bad = self.vision.leg_bad
            
            if back_bad or leg_bad:
                self.last_bad_time = time.time()
                if (time.time() - self.last_audio_time) > 4.0: 
                    if self.loaded_sounds:
                        random.choice(self.loaded_sounds).play()
                        self.last_audio_time = time.time()
            
            is_bad_memory = (time.time() - self.last_bad_time) < 1.0

            if is_bad_memory:
                self.good_posture_timer = 0
                self.bad_posture_timer += dt
                
                if self.bad_posture_timer > 3.0:
                    if self.current_state == self.STATE_PLAYING:
                        self.current_state = self.STATE_STRUGGLING
                        if self.final_fall_direction == 0: 
                            self.final_fall_direction = random.choice([-1, 1])
                    
                    self.instability += 35 * dt 
                    self.balance += self.final_fall_direction * 50 * dt
                
            else:
                self.bad_posture_timer = 0
                self.good_posture_timer += dt
                
                if self.good_posture_timer > 3.0:
                    self.current_state = self.STATE_PLAYING
                    self.instability = 0
                    self.balance = 0
                    self.velocity = 0
                    self.final_fall_direction = 0
                elif self.current_state == self.STATE_STRUGGLING:
                    self.instability = max(0, self.instability - 10 * dt)

            self.instability = max(0, min(self.instability, 100))
            
            if self.instability >= 100:
                self.current_state = self.STATE_GAMEOVER
                self.activate_ragdoll()

        rope_vibe = math.sin(time.time() * 50) * (self.instability / 20.0)
        s_dir = self.final_fall_direction if self.current_state == self.STATE_STRUGGLING else 0
        if self.current_state != self.STATE_GAMEOVER:
            self.player.calculate_positions(self.balance, self.instability, True, dt, self.rope_y + rope_vibe, s_dir)

    def draw(self):
        game_height = WINDOW_HEIGHT - BAR_HEIGHT
        game_surface = pygame.Surface((GAME_WIDTH, game_height))
        game_surface.blit(self.assets.sky_surf, (0,0))
        self.assets.draw_clouds(game_surface)
        vibe = math.sin(time.time() * 50) * (self.instability / 20.0)
        visual_rope_y = self.rope_y + vibe
        self.assets.draw_rope_horizontal(game_surface, self.rope_y, self.instability)

        if self.current_state in [self.STATE_PLAYING, self.STATE_STRUGGLING]:
            self.player.x = GAME_WIDTH // 2
            struggle_dir = 0
            if self.current_state == self.STATE_STRUGGLING:
                struggle_dir = self.final_fall_direction
                warn = self.big_font.render("¡¡CUIDADO!!", True, (255, 50, 0))
                game_surface.blit(warn, (GAME_WIDTH//2 - warn.get_width()//2, 200))
            
            elif self.bad_posture_timer > 0.5:
                charge_pct = min(1.0, self.bad_posture_timer / 3.0)
                radius = 70
                rect = pygame.Rect(self.player.x - radius, self.player.y - radius - 50, radius*2, radius*2)
                pygame.draw.arc(game_surface, (255, 200, 0), rect, 0, charge_pct * 6.28, 5)
                if charge_pct > 0.3:
                    warn = self.font.render("¡CORRIGE POSTURA!", True, (255, 200, 0))
                    game_surface.blit(warn, (GAME_WIDTH//2 - warn.get_width()//2, 200))

            self.player.draw(game_surface, self.balance, self.instability, True, 0.05, visual_rope_y, struggle_dir)

            elapsed_play = time.time() - self.play_start_time
            if self.current_state == self.STATE_PLAYING and elapsed_play < 5.0:
                countdown = int(6 - elapsed_play)
                count_txt = self.big_font.render(f"PREPÁRATE: {countdown}", True, (0, 255, 0))
                bg_rect = count_txt.get_rect(center=(GAME_WIDTH//2, 150))
                bg_surf = pygame.Surface((bg_rect.width + 20, bg_rect.height + 10), pygame.SRCALPHA)
                bg_surf.fill((0,0,0, 100))
                game_surface.blit(bg_surf, (bg_rect.x - 10, bg_rect.y - 5))
                game_surface.blit(count_txt, bg_rect)

            bar_w = 200
            pygame.draw.rect(game_surface, (20,20,40, 150), (20, 50, bar_w, 12), border_radius=5)
            fill = (self.instability / 100.0) * bar_w
            col = (50, 255, 150) if self.instability < 50 else (255, 80, 80)
            pygame.draw.rect(game_surface, col, (20, 50, fill, 12), border_radius=5)
            lbl = self.font.render("INESTABILIDAD", True, (220,230,255))
            game_surface.blit(lbl, (20, 25))

        elif self.current_state == self.STATE_GAMEOVER:
            if self.ragdoll: self.ragdoll.draw(game_surface)
            overlay = pygame.Surface((GAME_WIDTH, game_height), pygame.SRCALPHA)
            overlay.fill((0,0,0, 150))
            game_surface.blit(overlay, (0,0))
            txt = self.big_font.render("¡TE CAÍSTE!", True, (255, 50, 50))
            game_surface.blit(txt, (GAME_WIDTH//2 - txt.get_width()//2, 200))
            
            mx, my = pygame.mouse.get_pos(); mx -= CAM_WIDTH
            col = BUTTON_DANGER_HOVER if self.btn_restart_rect.collidepoint(mx, my) else BUTTON_DANGER
            pygame.draw.rect(game_surface, col, self.btn_restart_rect, border_radius=10)
            pygame.draw.rect(game_surface, (255,255,255), self.btn_restart_rect, 3, border_radius=10)
            
            if self.gesture_start_timer > 0.1:
                pct = min(1.0, self.gesture_start_timer / 2.0)
                rect = self.btn_restart_rect.inflate(20, 20)
                pygame.draw.rect(game_surface, (50, 255, 50), rect, 3, border_radius=15)
                fill_rect = pygame.Rect(rect.x, rect.bottom + 10, rect.width * pct, 5)
                pygame.draw.rect(game_surface, (50, 255, 50), fill_rect)

            btn_txt = self.font.render("REINICIAR", True, (255,255,255))
            game_surface.blit(btn_txt, (self.btn_restart_rect.centerx - btn_txt.get_width()//2, self.btn_restart_rect.centery - btn_txt.get_height()//2))
            info = self.font.render("Levanta la MANO o presiona [ESPACIO]", True, (200,200,200))
            game_surface.blit(info, (GAME_WIDTH//2 - info.get_width()//2, 500))

        elif self.current_state == self.STATE_MENU:
            self.player.x = GAME_WIDTH // 2
            self.player.draw(game_surface, 0, 0, False, 0.0, self.rope_y, 0)
            overlay = pygame.Surface((GAME_WIDTH, game_height), pygame.SRCALPHA)
            overlay.fill((0,0,0, 100))
            game_surface.blit(overlay, (0,0))
            title = self.big_font.render("SKY WALKER", True, (255,255,255))
            game_surface.blit(title, (GAME_WIDTH//2 - title.get_width()//2, 150))
            
            mx, my = pygame.mouse.get_pos(); mx -= CAM_WIDTH
            col = BUTTON_HOVER if self.btn_rect.collidepoint(mx, my) else BUTTON_COLOR
            pygame.draw.rect(game_surface, col, self.btn_rect, border_radius=10)
            pygame.draw.rect(game_surface, (255,255,255), self.btn_rect, 3, border_radius=10)
            
            if self.gesture_start_timer > 0.1:
                pct = min(1.0, self.gesture_start_timer / 2.0)
                rect = self.btn_rect.inflate(20, 20)
                pygame.draw.rect(game_surface, (50, 255, 50), rect, 3, border_radius=15)
                fill_rect = pygame.Rect(rect.x, rect.bottom + 10, rect.width * pct, 5)
                pygame.draw.rect(game_surface, (50, 255, 50), fill_rect)

            btn_txt = self.font.render("COMENZAR", True, (0,0,0))
            game_surface.blit(btn_txt, (self.btn_rect.centerx - btn_txt.get_width()//2, self.btn_rect.centery - btn_txt.get_height()//2))
            instr = self.font.render("Levanta la MANO o presiona [ESPACIO]", True, (200,200,200))
            game_surface.blit(instr, (GAME_WIDTH//2 - instr.get_width()//2, 450))

        if self.vision.latest_frame is not None:
            cam_surf = self.process_cam_for_display(self.vision.latest_frame)
            self.screen.blit(cam_surf, (0, 0))
        else:
            cam_surf = pygame.Surface((CAM_WIDTH, WINDOW_HEIGHT - BAR_HEIGHT))
            font = pygame.font.SysFont("Arial", 30)
            txt = font.render("CÁMARA...", True, (100, 255, 100))
            cam_surf.blit(txt, (50, (WINDOW_HEIGHT - BAR_HEIGHT)//2))
            self.screen.blit(cam_surf, (0, 0))

        self.bottom_bar_surf.fill(BAR_BG_COLOR)
        pygame.draw.line(self.bottom_bar_surf, (100, 100, 100), (0, 0), (WINDOW_WIDTH, 0), 2)
        
        angle_color_rgb = (255, 0, 0) if self.vision.leg_bad else (0, 255, 0)
        pygame.draw.circle(self.bottom_bar_surf, angle_color_rgb, (50, BAR_HEIGHT // 2), 20, 3)
        angle_text = f"{int(self.vision.leg_angle)}°"
        txt_surf = self.font.render(angle_text, True, (255, 255, 255))
        txt_rect = txt_surf.get_rect(center=(50, BAR_HEIGHT // 2))
        self.bottom_bar_surf.blit(txt_surf, txt_rect)
        label_surf = self.font.render("Ángulo Rodilla", True, (200, 200, 200))
        self.bottom_bar_surf.blit(label_surf, (85, BAR_HEIGHT // 2 - label_surf.get_height() // 2))

        score_text = f"TIEMPO: {self.current_score:.1f}s"
        score_surf = self.font.render(score_text, True, (255, 255, 0))
        score_rect = score_surf.get_rect(center=(WINDOW_WIDTH//2, BAR_HEIGHT//2))
        self.bottom_bar_surf.blit(score_surf, score_rect)

        best_text = f"MEJOR: {self.best_score:.1f}s"
        best_surf = self.font.render(best_text, True, (0, 255, 255))
        best_rect = best_surf.get_rect(center=(WINDOW_WIDTH - 150, BAR_HEIGHT//2))
        self.bottom_bar_surf.blit(best_surf, best_rect)

        self.screen.blit(game_surface, (CAM_WIDTH, 0))
        self.screen.blit(self.bottom_bar_surf, (0, WINDOW_HEIGHT - BAR_HEIGHT))
        pygame.display.flip()

    def process_cam_for_display(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.transpose(frame, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        surf = pygame.transform.flip(surf, True, False)
        surf = pygame.transform.scale(surf, (CAM_WIDTH, WINDOW_HEIGHT - BAR_HEIGHT))
        return surf

    def run(self):
        running = True
        while running:
            dt = self.clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = pygame.mouse.get_pos(); mx -= CAM_WIDTH
                    if self.current_state == self.STATE_MENU:
                        if self.btn_rect.collidepoint(mx, my):
                            self.current_state = self.STATE_PLAYING
                            self.play_start_time = time.time()
                    elif self.current_state == self.STATE_GAMEOVER:
                        if self.btn_restart_rect.collidepoint(mx, my): self.reset_game()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q: running = False
                    if self.current_state == self.STATE_MENU:
                        if event.key == pygame.K_SPACE:
                            self.current_state = self.STATE_PLAYING
                            self.play_start_time = time.time()
                    elif self.current_state == self.STATE_GAMEOVER:
                        if event.key == pygame.K_SPACE or event.key == pygame.K_r: self.reset_game()
            self.update(dt)
            self.draw()
        self.vision.running = False
        pygame.quit()

if __name__ == "__main__":
    game = SmoothGame()
    game.run()