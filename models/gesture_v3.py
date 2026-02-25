"""
Weekly diaries: 
SOVA Gesture Recognition V6 - Production Ready
- Fewer false positives (gestures must persist)
- Better multi-person detection
- Improved YOLO phone detection
- Longer cooldowns, more realistic
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
import pyttsx3
import mss
import mss.tools
from PIL import Image

class GestureRecognizer:
    def __init__(self, enable_speech=True, capture_mode='screen', debug_mode=False):
        print("🚀 Initializing SOVA ")
        
        self.capture_mode = capture_mode
        self.debug_mode = debug_mode
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # YOLO for phone detection
        self.use_phone_detection = False
        try:
            from ultralytics import YOLO
            self.phone_detector = YOLO('yolov8n.pt')
            self.use_phone_detection = True
            print("✅ YOLO phone detection enabled")
        except Exception as e:
            print(f"⚠️ YOLO unavailable: {e}")
        
        # Hand detection - track up to 4 hands (2 people)
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=4,  # Support 2 people
            min_detection_confidence=0.75,  # Higher threshold
            min_tracking_confidence=0.6
        )
        
        # Pose detection
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        
        # Track multiple people
        self.previous_person_count = 0
        self.person_history = deque(maxlen=10)
        
        # Text-to-Speech
        self.enable_speech = enable_speech
        if enable_speech:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 150)
                self.tts_engine.setProperty('volume', 0.9)
                print("✅ Text-to-Speech enabled")
            except:
                print("⚠️ TTS unavailable")
                self.enable_speech = False
        
        # GESTURE PERSISTENCE: Gesture must appear for N frames before reporting
        self.gesture_persistence_buffer = {}  # {gesture_id: [frame_count, gesture_data]}
        self.PERSISTENCE_THRESHOLD = 7  # Must appear in 7 consecutive frames
        
        # Tracking variables
        self.hand_position_history = {}  # Track per hand for wave detection
        self.last_gesture_time = {}
        
        # REALISTIC COOLDOWNS (in seconds)
        self.gesture_cooldown = 10.0  # 10 seconds between SAME gesture
        self.global_speech_cooldown = 5.0  # 5 seconds between ANY announcement
        self.last_speech_time = 0
        self.last_announcement = ""
        
        # Display buffer
        self.display_text_buffer = []
        self.display_duration = 4.0
        
        # Screen capture
        if capture_mode == 'screen':
            self.sct = mss.mss()
        
        print("✅ SOVA V6 Ready - Production Mode")
        
    def speak(self, text):
        """Text-to-speech with cooldown"""
        current_time = time.time()
        
        if self.debug_mode:
            print(f"🔍 DEBUG: {text}")
            return
        
        if current_time - self.last_speech_time < self.global_speech_cooldown:
            return
        
        if self.enable_speech and text != self.last_announcement:
            print(f"🔊 {text}")
            try:
                self.last_speech_time = current_time
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
                estimated_duration = len(text) * 0.05
                self.last_speech_time = time.time() + estimated_duration
            except:
                print(f"📝 {text}")
            self.last_announcement = text
    
    def get_finger_distance(self, point1, point2):
        """Calculate distance between two points"""
        return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def is_finger_extended(self, tip, pip, mcp, wrist):
        """Check if finger is extended (orientation-independent)"""
        tip_dist = self.get_finger_distance(wrist, tip)
        pip_dist = self.get_finger_distance(wrist, pip)
        return tip_dist > pip_dist * 1.12  # Stricter threshold
    
    def is_thumb_extended(self, thumb_tip, thumb_ip, index_mcp, wrist):
        """Check if thumb is extended"""
        thumb_to_index = self.get_finger_distance(thumb_tip, index_mcp)
        ip_to_index = self.get_finger_distance(thumb_ip, index_mcp)
        return thumb_to_index > ip_to_index * 1.2  # Stricter
    
    def detect_hand_gesture(self, hand_landmarks, hand_id):
        """V6 - Stricter gesture detection with persistence"""
        lm = hand_landmarks.landmark
        
        # Key landmarks
        wrist = lm[0]
        thumb_tip = lm[4]
        thumb_ip = lm[3]
        thumb_mcp = lm[2]
        index_tip = lm[8]
        index_pip = lm[6]
        index_mcp = lm[5]
        middle_tip = lm[12]
        middle_pip = lm[10]
        middle_mcp = lm[9]
        ring_tip = lm[16]
        ring_pip = lm[14]
        ring_mcp = lm[13]
        pinky_tip = lm[20]
        pinky_pip = lm[18]
        pinky_mcp = lm[17]
        
        # Check extended fingers
        thumb_extended = self.is_thumb_extended(thumb_tip, thumb_ip, index_mcp, wrist)
        index_extended = self.is_finger_extended(index_tip, index_pip, index_mcp, wrist)
        middle_extended = self.is_finger_extended(middle_tip, middle_pip, middle_mcp, wrist)
        ring_extended = self.is_finger_extended(ring_tip, ring_pip, ring_mcp, wrist)
        pinky_extended = self.is_finger_extended(pinky_tip, pinky_pip, pinky_mcp, wrist)
        
        num_extended = sum([thumb_extended, index_extended, middle_extended, 
                           ring_extended, pinky_extended])
        
        # Track hand position for wave detection
        if hand_id not in self.hand_position_history:
            self.hand_position_history[hand_id] = deque(maxlen=12)
        self.hand_position_history[hand_id].append((wrist.x, wrist.y))
        
        # ==================== GESTURE DETECTION ====================
        
        # 1. THUMBS UP - Very strict
        if thumb_extended and not index_extended and not middle_extended and \
           not ring_extended and not pinky_extended:
            thumb_vertical = abs(thumb_tip.y - thumb_mcp.y)
            # Must be pointing UP (thumb tip above mcp)
            if thumb_vertical > 0.10 and thumb_tip.y < thumb_mcp.y:
                return "thumbs_up", 0.90
        
        # 2. THUMBS DOWN - Very strict
        if thumb_extended and not index_extended and not middle_extended and \
           not ring_extended and not pinky_extended:
            thumb_vertical = abs(thumb_tip.y - thumb_mcp.y)
            # Must be pointing DOWN (thumb tip below mcp)
            if thumb_vertical > 0.10 and thumb_tip.y > thumb_mcp.y:
                return "thumbs_down", 0.90
        
        # 3. POINTING - Must be clearly pointing
        if index_extended and not middle_extended and not ring_extended and not pinky_extended:
            index_length = self.get_finger_distance(index_tip, index_mcp)
            
            # Index must be prominently extended
            if index_length > 0.15:
                # Not a thumbs up confused as pointing
                if not thumb_extended or index_length > self.get_finger_distance(thumb_tip, thumb_mcp) * 1.2:
                    return "pointing", 0.92
        
        # 4. PEACE SIGN - Strict V-shape
        if index_extended and middle_extended and not ring_extended and not pinky_extended:
            finger_spread = self.get_finger_distance(index_tip, middle_tip)
            
            if finger_spread > 0.10:  # Wide V
                index_length = self.get_finger_distance(index_tip, index_mcp)
                middle_length = self.get_finger_distance(middle_tip, middle_mcp)
                
                # Both fingers well extended
                if index_length > 0.13 and middle_length > 0.13:
                    return "peace_sign", 0.94
        
        # 5. WAVING - Motion-based
        if num_extended >= 4:
            if self.detect_wave_motion(hand_id):
                return "waving", 0.93
            # Removed open_palm - too noisy
        
        # 6. HAND RAISED - High in frame
        if num_extended >= 4 and wrist.y < 0.4:  # Higher threshold
            if self.detect_wave_motion(hand_id):
                return "waving", 0.93
            return "hand_raised", 0.89
        
        # 7. OK SIGN
        thumb_index_dist = self.get_finger_distance(thumb_tip, index_tip)
        if thumb_index_dist < 0.04 and middle_extended and ring_extended:
            return "ok_sign", 0.90
        
        return "unknown", 0.0
    
    def detect_wave_motion(self, hand_id):
        """Detect waving motion for specific hand"""
        if hand_id not in self.hand_position_history:
            return False
        
        history = self.hand_position_history[hand_id]
        if len(history) < 10:
            return False
        
        x_positions = [pos[0] for pos in list(history)[-10:]]
        x_diff = np.diff(x_positions)
        
        direction_changes = 0
        for i in range(len(x_diff) - 1):
            if x_diff[i] * x_diff[i+1] < 0:
                direction_changes += 1
        
        # Stricter: need at least 3 direction changes
        return direction_changes >= 3
    
    def detect_phone_usage(self, frame):
        """YOLO-based phone detection"""
        
        if not self.use_phone_detection:
            return False, 0.0
        
        try:
            # Run YOLO inference
            results = self.phone_detector(frame, verbose=False, conf=0.55)
            
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Class 67 = cell phone (COCO dataset)
                    if class_id == 67 and confidence > 0.55:
                        
                        # Draw box in debug mode
                        if self.debug_mode:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            cv2.rectangle(frame, 
                                        (int(x1), int(y1)), 
                                        (int(x2), int(y2)), 
                                        (0, 0, 255), 3)
                            cv2.putText(frame, f"PHONE {confidence:.0%}", 
                                      (int(x1), int(y1) - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        return True, confidence
        except Exception as e:
            if self.debug_mode:
                print(f"YOLO error: {e}")
        
        return False, 0.0
    
    def detect_person_status(self, pose_landmarks, frame_shape):
        """Detect if person is leaving frame"""
        h, w, _ = frame_shape
        
        if pose_landmarks is None:
            return "person_left_frame", 0.95
        
        nose = pose_landmarks.landmark[0]
        
        if nose.x < 0.12 or nose.x > 0.88:
            return "person_leaving_frame", 0.88
        
        if nose.y < 0.08 or nose.y > 0.92:
            return "person_leaving_frame", 0.88
        
        return "person_present", 0.90
    
    def detect_distraction(self, pose_landmarks):
        """Detect if person is looking away"""
        if pose_landmarks is None:
            return False, 0.0
        
        left_ear = pose_landmarks.landmark[7]
        right_ear = pose_landmarks.landmark[8]
        
        avg_visibility = (left_ear.visibility + right_ear.visibility) / 2
        
        if avg_visibility < 0.45:
            return True, 0.78
        
        ear_x_diff = abs(left_ear.x - right_ear.x)
        if ear_x_diff < 0.04:
            return True, 0.75
        
        return False, 0.0
    def detect_face_presence(self, pose_landmarks):
        """Return True if a real face is visible (eyes + mouth visible)."""
        if pose_landmarks is None:
            return False

        # Key facial landmarks
        left_eye = pose_landmarks.landmark[2]
        right_eye = pose_landmarks.landmark[5]
        mouth_left = pose_landmarks.landmark[9]
        mouth_right = pose_landmarks.landmark[10]

        # Visibility threshold
        vis = [
            left_eye.visibility,
            right_eye.visibility,
            mouth_left.visibility,
            mouth_right.visibility
        ]

        # If at least 2 facial points are clearly visible → person is present
        return sum(v > 0.5 for v in vis) >= 2
    def assign_hands_to_person(self, hand_results, pose_landmarks):
        """
        Returns a dict: {hand_id: person_label}
        Uses pose landmarks (shoulders) to group hands to the same person.
        """
        if pose_landmarks is None:
            # fallback: left/right split
            mapping = {}
            for idx, hand in enumerate(hand_results.multi_hand_landmarks):
                wrist_x = hand.landmark[0].x
                if wrist_x < 0.5:
                    mapping[idx] = "Person"
                else:
                    mapping[idx] = "Person"
            return mapping

        # Use shoulders to determine center of person
        left_shoulder = pose_landmarks.landmark[11].x
        right_shoulder = pose_landmarks.landmark[12].x
        center = (left_shoulder + right_shoulder) / 2

        mapping = {}
        for idx, hand in enumerate(hand_results.multi_hand_landmarks):
            wrist_x = hand.landmark[0].x
            # If wrist is near the shoulder center → same person
            if abs(wrist_x - center) < 0.25:
                mapping[idx] = "Person"
            else:
                mapping[idx] = "Unknown"
        return mapping

    def check_gesture_persistence(self, gesture_id, gesture_type, confidence):
        """
        Check if gesture has appeared consistently
        Returns True if gesture should be reported
        """
        current_time = time.time()
        
        if gesture_id not in self.gesture_persistence_buffer:
            self.gesture_persistence_buffer[gesture_id] = {
                'count': 1,
                'type': gesture_type,
                'confidence': confidence,
                'first_seen': current_time
            }
            return False
        
        buffer = self.gesture_persistence_buffer[gesture_id]
        
        # Same gesture continuing
        if buffer['type'] == gesture_type:
            buffer['count'] += 1
            buffer['confidence'] = max(buffer['confidence'], confidence)
            
            # Gesture persisted long enough
            if buffer['count'] >= self.PERSISTENCE_THRESHOLD:
                # Reset buffer after reporting
                del self.gesture_persistence_buffer[gesture_id]
                return True
        else:
            # Gesture changed, reset
            self.gesture_persistence_buffer[gesture_id] = {
                'count': 1,
                'type': gesture_type,
                'confidence': confidence,
                'first_seen': current_time
            }
        
        return False
    
    def should_report(self, detection_type):
        """Check cooldown before reporting"""
        current_time = time.time()
        
        if detection_type in self.last_gesture_time:
            time_since_last = current_time - self.last_gesture_time[detection_type]
            if time_since_last < self.gesture_cooldown:
                return False
        
        self.last_gesture_time[detection_type] = current_time
        return True
    
    def draw_gesture_box(self, frame, hand_landmarks, gesture_type, confidence):
        """Draw bounding box around gesture"""
        h, w, _ = frame.shape
        
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]
        
        x_min = int(min(x_coords)) - 25
        y_min = int(min(y_coords)) - 25
        x_max = int(max(x_coords)) + 25
        y_max = int(max(y_coords)) + 25
        
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)
        
        # Color based on confidence
        if confidence >= 0.92:
            color = (0, 255, 0)  # Green - very high
        elif confidence >= 0.87:
            color = (0, 255, 255)  # Yellow - high
        else:
            color = (0, 165, 255)  # Orange - medium
        
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 3)
        
        label = f"{gesture_type.upper()} {confidence:.0%}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        
        cv2.rectangle(frame, 
                      (x_min, y_min - label_size[1] - 15),
                      (x_min + label_size[0] + 10, y_min),
                      color, -1)
        
        cv2.putText(frame, label, (x_min + 5, y_min - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    def generate_multi_person_description(self, gesture_groups):
        """
        Generate smart descriptions for multiple people
        gesture_groups: {gesture_type: [person_labels]}
        """
        descriptions = []
        
        gesture_names = {
            'thumbs_up': "thumbs up",
            'thumbs_down': "thumbs down",
            'peace_sign': "peace sign",
            'waving': "waving",
            'pointing': "pointing",
            'hand_raised': "hand raised",
            'ok_sign': "OK sign"
        }
        
        for gesture_type, people in gesture_groups.items():
            if gesture_type not in gesture_names:
                continue
            
            gesture_name = gesture_names[gesture_type]
            
            if len(people) == 1:
                # Single person
                descriptions.append(f"{people[0]} {gesture_name}")
            elif len(people) == 2:
                # Both people doing same gesture
                descriptions.append(f"Both people {gesture_name}")
            else:
                # Multiple people
                descriptions.append(f"Multiple people {gesture_name}")
        
        return ". ".join(descriptions)
    
    def process_frame(self, frame):
        """Main processing function - V6 Production"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        
        observations = {
            'gestures': [],
            'body_actions': [],
            'timestamp': time.time(),
            'description': ""
        }
        
        # Hand detection
        hand_results = self.hands.process(rgb_frame)
        
        # Track gestures by type for multi-person grouping
        gesture_groups = {}  # {gesture_type: [person_labels]}
        
        if hand_results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                gesture_type, confidence = self.detect_hand_gesture(hand_landmarks, idx)
                
                # Determine person label based on position
                wrist_x = hand_landmarks.landmark[0].x
                if wrist_x < 0.33:
                    person_label = "Person on left"
                elif wrist_x > 0.67:
                    person_label = "Person on right"
                else:
                    person_label = "Person in center"
                
                # High confidence threshold
                CONFIDENCE_THRESHOLD = 0.88
                
                if gesture_type != "unknown" and confidence >= CONFIDENCE_THRESHOLD:
                    
                    # Check persistence (must appear in N consecutive frames)
                    gesture_id = f"{gesture_type}_{idx}"
                    
                    if self.check_gesture_persistence(gesture_id, gesture_type, confidence):
                        
                        # Check cooldown
                        cooldown_id = f"{gesture_type}_{person_label}"
                        if self.should_report(cooldown_id):
                            
                            # Group by gesture type
                            if gesture_type not in gesture_groups:
                                gesture_groups[gesture_type] = []
                            gesture_groups[gesture_type].append(person_label)
                            
                            observations['gestures'].append({
                                'type': gesture_type,
                                'confidence': confidence,
                                'person': person_label,
                                'hand_id': idx
                            })
                
                # DEBUG MODE: Draw boxes
                if self.debug_mode and gesture_type != "unknown" and confidence >= 0.80:
                    self.draw_gesture_box(frame, hand_landmarks, gesture_type, confidence)
                
                # NORMAL MODE: Draw landmarks
                if not self.debug_mode:
                    self.mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                    )
        
        # Pose detection
        pose_results = self.pose.process(rgb_frame)
        
        # Count people
        num_hands = len(hand_results.multi_hand_landmarks) if hand_results.multi_hand_landmarks else 0
        # FACE-BASED PERSON DETECTION
        face_present = self.detect_face_presence(pose_results.pose_landmarks)

        if face_present:
            estimated_people = 1
        else:
            # fallback: hands only (less reliable)
            estimated_people = (num_hands + 1) // 2

        
        self.person_history.append(estimated_people)
        
        # Detect person changes (smoothed)
        if len(self.person_history) >= 5:
            avg_people = sum(list(self.person_history)[-5:]) / 5
            
            if avg_people > self.previous_person_count + 0.5:
                if self.should_report("person_approaching"):
                    observations['body_actions'].append("person_approaching")
            elif avg_people < self.previous_person_count - 0.5:
                if self.should_report("person_left"):
                    observations['body_actions'].append("person_left")
        
        self.previous_person_count = estimated_people
        
        # Phone detection (YOLO)
        using_phone, phone_conf = self.detect_phone_usage(frame)
        if using_phone and self.should_report("using_phone"):
            observations['body_actions'].append("using_phone")
        
        # Pose analysis
        if pose_results.pose_landmarks:
            person_status, status_conf = self.detect_person_status(
                pose_results.pose_landmarks, 
                frame.shape
            )
            
            if person_status != "person_present" and self.should_report(person_status):
                observations['body_actions'].append(person_status)
            
            is_distracted, distract_conf = self.detect_distraction(pose_results.pose_landmarks)
            if is_distracted and self.should_report("looking_away"):
                observations['body_actions'].append("looking_away")
            
            if not self.debug_mode:
                self.mp_drawing.draw_landmarks(
                    frame, 
                    pose_results.pose_landmarks, 
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2)
                )
        else:
            if self.should_report("person_left_frame"):
                observations['body_actions'].append("person_left_frame")
        
        # Generate smart multi-person descriptions
        gesture_desc = self.generate_multi_person_description(gesture_groups)
        
        action_phrases = {
            'person_left_frame': "Person left the view",
            'person_leaving_frame': "Person moving out of frame",
            'using_phone': "Person using phone",
            'looking_away': "Person looking away",
            'person_approaching': "Someone approaching from behind",
            'person_left': "Someone left"
        }
        
        action_descs = [action_phrases[a] for a in observations['body_actions'] 
                       if a in action_phrases]
        
        all_descriptions = []
        if gesture_desc:
            all_descriptions.append(gesture_desc)
        all_descriptions.extend(action_descs)
        
        description = ". ".join(all_descriptions)
        observations['description'] = description
        
        if description:
            self.speak(description)
        
        return observations, frame


def select_capture_region():
    """Select capture region"""
    print("\n" + "="*60)
    print("📺 SCREEN REGION SELECTION")
    print("="*60)
    print("\nOptions:")
    print("1. Full screen")
    print("2. Meeting window")
    print("3. Webcam (testing)")
    
    choice = input("\nChoice (1-3): ").strip()
    
    if choice == '3':
        return 'webcam', None
    
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        return 'screen' if choice == '1' else 'screen', monitor


def main():
    """Main loop - V6 Production"""
    print("\n" + "="*60)
    print("🎯 SOVA V6 - PRODUCTION READY")
    print("="*60)
    
    capture_type, region = select_capture_region()
    
    print("\n🔧 Debug mode?")
    debug_choice = input("Enable (y/n): ").strip().lower()
    debug_mode = debug_choice == 'y'
    
    recognizer = GestureRecognizer(
        enable_speech=not debug_mode,
        capture_mode=capture_type,
        debug_mode=debug_mode
    )
    
    if debug_mode:
        print("✅ DEBUG MODE")
    
    if capture_type == 'webcam':
        print("\n📹 Opening webcam...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Camera error!")
            return
        
        print("\n✅ Active!")
        print("\n📋 CONTROLS:")
        print("   Q - Quit")
        print("   S - Toggle speech")
        print("\n🎯 V6 FEATURES:")
        print("   • Gestures must persist (less false positives)")
        print("   • 15 second cooldown (realistic)")
        print("   • Multi-person: 'Both waving'")
        print("   • YOLO phone detection")
        
        fps_start = time.time()
        fps_count = 0
        fps = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            observations, annotated_frame = recognizer.process_frame(frame)
            
            fps_count += 1
            if (time.time() - fps_start) > 1:
                fps = fps_count
                fps_count = 0
                fps_start = time.time()
            
            current_time = time.time()
            
            for gesture in observations['gestures']:
                text = f"✋ {gesture['type'].upper()} ({gesture['confidence']:.0%})"
                recognizer.display_text_buffer.append({
                    'text': text,
                    'time': current_time,
                    'color': (0, 255, 0)
                })
            
            for action in observations['body_actions']:
                text = f"👤 {action.replace('_', ' ').upper()}"
                recognizer.display_text_buffer.append({
                    'text': text,
                    'time': current_time,
                    'color': (255, 100, 0)
                })
            
            recognizer.display_text_buffer = [
                item for item in recognizer.display_text_buffer 
                if current_time - item['time'] < recognizer.display_duration
            ]
            
            mode_text = "DEBUG" if debug_mode else "LIVE"
            mode_color = (255, 255, 0) if debug_mode else (0, 255, 0)
            cv2.putText(annotated_frame, mode_text, 
                       (annotated_frame.shape[1] - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
            
            cv2.putText(annotated_frame, f"FPS: {fps}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if not debug_mode:
                y = 60
                for item in recognizer.display_text_buffer:
                    cv2.putText(annotated_frame, item['text'], (10, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, item['color'], 2)
                    y += 35
            
            cv2.imshow('SOVA V6', annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                recognizer.enable_speech = not recognizer.enable_speech
                print(f"🔊 Speech: {'ON' if recognizer.enable_speech else 'OFF'}")
        
        cap.release()
        cv2.destroyAllWindows()
    
    else:
        # Screen capture mode
        print("\n✅ Screen capture active!")
        
        with mss.mss() as sct:
            while True:
                screenshot = sct.grab(region)
                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
                observations, annotated_frame = recognizer.process_frame(frame)
                
                display_frame = cv2.resize(annotated_frame, 
                                          (annotated_frame.shape[1] // 2, 
                                           annotated_frame.shape[0] // 2))
                
                cv2.imshow('SOVA V6', display_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

#nodding and shaking head
#model learning from the video feedback itself and based on if the user allows for data to be saved 
