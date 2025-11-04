#!/usr/bin/env python3
"""
Voice Assistant optimized for NVIDIA Jetson Nano - Office Receptionist Persona
FIXED GUI VERSION - Proper initialization order and rendering
"""
import os
import sys
import subprocess
import time
import ctypes
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor
import atexit
import re
import random

# ---------------------------------------------------------------------
# GUI/Audio Fix: Set drivers before imports
# ---------------------------------------------------------------------
os.environ['SDL_VIDEODRIVER'] = 'x11'
os.environ['SDL_AUDIODRIVER'] = 'alsa'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import pygame

# ---------------------------------------------------------------------
# Locate the build directory
# ---------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(ROOT_DIR, "build")
if BUILD_DIR not in sys.path:
    sys.path.append(BUILD_DIR)

# Import face recognition
try:
    import face_recog_bind
    FACE_RECOG_AVAILABLE = True
    print("🧩 face_recog_bind loaded successfully.")
    db_size = face_recog_bind.get_database_size()
    print(f"📸 Face database: {db_size} faces loaded and ready")
    atexit.register(face_recog_bind.cleanup_system)
except Exception as e:
    FACE_RECOG_AVAILABLE = False
    face_recog_bind = None
    print(f"⚠️ face_recog_bind not available: {e}")

# ---------------------------------------------------------------------
# Suppress ALSA errors
# ---------------------------------------------------------------------
ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
                                      ctypes.c_int, ctypes.c_char_p)
def py_error_handler(filename, line, function, err, fmt): pass
c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
try:
    asound = ctypes.cdll.LoadLibrary('libasound.so.2')
    asound.snd_lib_error_set_handler(c_error_handler)
except:
    pass

os.environ['PYTHONWARNINGS'] = 'ignore'
import speech_recognition as sr
from gtts import gTTS
from openai import OpenAI
warnings.filterwarnings("ignore")

# Try to import Vosk
VOSK_AVAILABLE = False
try:
    from vosk import Model, KaldiRecognizer
    import json
    import wave
    VOSK_AVAILABLE = True
    print("✅ Vosk available for offline speech recognition")
except ImportError:
    print("ℹ️ Vosk not installed. Using Google STT (online)")

# ---------------------------------------------------------------------
# Text Cleaning for Natural Speech
# ---------------------------------------------------------------------
def clean_text_for_speech(text):
    """Remove markdown, special characters, and format text naturally for speech"""
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'\1', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    text = re.sub(r'_(.+?)_', r'\1', text)
    text = re.sub(r'#{1,6}\s+', '', text)
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'`(.+?)`', r'\1', text)
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'link', text)
    text = text.replace('&', 'and').replace('@', 'at').replace('#', 'number')
    text = text.replace('%', 'percent').replace('$', 'dollars')
    text = text.replace('>', 'greater than').replace('<', 'less than')
    text = text.replace('=', 'equals').replace('+', 'plus')
    text = text.replace('|', '').replace('~', '').replace('^', '')
    text = text.replace('[', '').replace(']', '').replace('{', '').replace('}', '')
    text = re.sub(r'\n+', '. ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\.{2,}', '.', text)
    return text.strip()

# ---------------------------------------------------------------------
# Main Assistant Class
# ---------------------------------------------------------------------
class JetsonVoiceAssistant:
    def __init__(self, debug=False):
        self.debug = debug
        self.model = "nvidia/nemotron-nano-12b-v2-vl:free"
        api_key = "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"
        if not api_key:
            print("\n" + "="*60)
            print("🚨 CRITICAL: OPENROUTER_API_KEY not set!")
            print("To fix: Run 'export OPENROUTER_API_KEY=sk-or-v1-your_actual_key_here'")
            print("Get a FREE key: https://openrouter.ai/keys")
            print("="*60 + "\n")
            api_key = "dummy"
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        
        self.system_prompt = """You are a professional office receptionist assistant.
Your personality:
- Polite, professional, and efficient
- Clear and concise in communication
- Helpful but businesslike
- You greet visitors, answer questions about the office, schedule meetings, and provide directions
- You maintain a warm but professional tone

Important instructions:
- Keep responses SHORT and CLEAR (1-3 sentences maximum)
- Use simple, natural language - NO markdown, NO special formatting, NO asterisks, NO bullets
- Speak naturally as if talking to someone in person
- For lists, say "first", "second", "third" instead of using numbers or bullets
- Always be respectful and courteous"""

        # Initialize GUI state variables BEFORE any pygame operations
        self.state = 'idle'
        self.user_text = ""
        self.assistant_text = ""
        self.blink_timer = 0
        self.is_blinking = False
        self.eye_offset_x = 0
        self.eye_offset_y = 0
        self.mouth_open = 0
        
        # Speech recognizer setup
        self.recognizer = sr.Recognizer()
        self._configure_recognizer()
        
        # Initialize ThreadPoolExecutor
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Initialize Pygame completely FIRST
        self._init_pygame()
        
        # Then setup audio
        self._setup_jetson_audio()
        
        # Initialize Vosk if available
        self.vosk_model = None
        if VOSK_AVAILABLE:
            self._init_vosk()
        
        # Pre-cache audio files
        self.audio_cache = {}
        self._precache_common_phrases()
        
        print("\n✅ Office Receptionist Assistant Ready!\n")

    def _init_pygame(self):
        """Initialize Pygame completely - FIXED ORDER"""
        try:
            print("🎮 Initializing Pygame...")
            
            # Initialize ALL pygame modules
            pygame.init()
            
            # Set up display with specific flags
            self.screen = pygame.display.set_mode((800, 600), pygame.HWSURFACE | pygame.DOUBLEBUF)
            pygame.display.set_caption("Office Receptionist Assistant")
            
            # Initialize clock
            self.clock = pygame.time.Clock()
            
            # Initialize fonts AFTER pygame.init()
            pygame.font.init()
            self.font = pygame.font.SysFont('arial', 20, bold=False)
            self.small_font = pygame.font.SysFont('arial', 16, bold=False)
            
            # Colors
            self.bg_color = (135, 206, 235)  # Sky blue
            
            # Fill with background immediately
            self.screen.fill(self.bg_color)
            pygame.display.flip()
            
            print("✅ Pygame initialized successfully")
            
            # Draw initial face
            time.sleep(0.1)  # Small delay for display to settle
            self.draw_face()
            pygame.display.flip()
            
            print("🖥️ GUI ready with cute face!")
            
        except Exception as e:
            print(f"❌ GUI initialization failed: {e}")
            import traceback
            traceback.print_exc()
            self.screen = None

    def _setup_jetson_audio(self):
        """Setup audio after display"""
        try:
            if self.debug:
                self._print_audio_devices()
            self._test_microphone()
        except Exception as e:
            print(f"⚠️ Audio setup warning: {e}")

    def draw_face(self):
        """Draw professional cartoon-style animated face"""
        if not self.screen:
            return
        
        try:
            # Clear screen with gradient-like background
            self.screen.fill(self.bg_color)
            
            # Head center and size
            head_center = (400, 280)
            head_width = 140
            head_height = 160
            
            # Neck/Shoulders with collar
            neck_color = (255, 235, 205)
            pygame.draw.rect(self.screen, neck_color, (head_center[0] - 30, head_center[1] + 100, 60, 80))
            
            # Professional attire (yellow blazer/top)
            blazer_color = (255, 200, 50)
            pygame.draw.polygon(self.screen, blazer_color, [
                (head_center[0] - 80, head_center[1] + 150),
                (head_center[0] + 80, head_center[1] + 150),
                (head_center[0] + 100, 500),
                (head_center[0] - 100, 500)
            ])
            
            # White collar
            collar_points = [
                (head_center[0] - 25, head_center[1] + 120),
                (head_center[0] - 15, head_center[1] + 140),
                (head_center[0], head_center[1] + 135),
                (head_center[0] + 15, head_center[1] + 140),
                (head_center[0] + 25, head_center[1] + 120)
            ]
            pygame.draw.polygon(self.screen, (255, 255, 255), collar_points)
            
            # Shadow for depth
            shadow_offset = 8
            pygame.draw.ellipse(self.screen, (210, 180, 140), 
                              (head_center[0] - head_width + shadow_offset, 
                               head_center[1] - head_height + shadow_offset, 
                               head_width * 2, head_height * 2))
            
            # Head (smooth ellipse)
            pygame.draw.ellipse(self.screen, (255, 224, 189), 
                              (head_center[0] - head_width, head_center[1] - head_height, 
                               head_width * 2, head_height * 2))
            
            # Hair - flowing brown hair
            hair_color = (101, 67, 33)
            # Left side hair
            pygame.draw.ellipse(self.screen, hair_color,
                              (head_center[0] - head_width - 20, head_center[1] - head_height - 10,
                               60, 180))
            # Right side hair
            pygame.draw.ellipse(self.screen, hair_color,
                              (head_center[0] + head_width - 40, head_center[1] - head_height - 10,
                               60, 180))
            # Top hair
            pygame.draw.ellipse(self.screen, hair_color,
                              (head_center[0] - head_width + 10, head_center[1] - head_height - 30,
                               head_width * 2 - 20, 100))
            
            # Hair highlights for dimension
            highlight_color = (139, 90, 43)
            pygame.draw.ellipse(self.screen, highlight_color,
                              (head_center[0] - 40, head_center[1] - head_height - 15,
                               80, 60))
            
            # Rosy cheeks
            blush_color = (255, 182, 193)
            pygame.draw.ellipse(self.screen, blush_color, 
                              (head_center[0] - 80, head_center[1] + 10, 35, 25))
            pygame.draw.ellipse(self.screen, blush_color, 
                              (head_center[0] + 45, head_center[1] + 10, 35, 25))
            
            # Eyes - Large expressive cartoon eyes
            eye_y = head_center[1] - 20
            left_eye_x = head_center[0] - 45
            right_eye_x = head_center[0] + 45
            
            # Blink logic
            self.blink_timer += 1 / 30
            if self.blink_timer > random.uniform(3, 6) and random.random() < 0.003:
                self.blink_timer = 0
                self.is_blinking = True
            
            if self.is_blinking:
                # Closed eyes - curved lashes
                for x in [left_eye_x, right_eye_x]:
                    pygame.draw.arc(self.screen, (0, 0, 0), 
                                  (x - 35, eye_y - 5, 70, 20), 0, 3.14, 4)
                    # Eyelashes
                    for i in range(-2, 3):
                        lash_x = x + i * 12
                        pygame.draw.line(self.screen, (0, 0, 0),
                                       (lash_x, eye_y), (lash_x - 3, eye_y - 8), 2)
                
                if self.blink_timer > 0.15:
                    self.is_blinking = False
            else:
                # Open eyes - much larger and more expressive
                eye_width = 42
                eye_height = 55
                
                if self.state == 'listening':
                    eye_height = 65  # Wide eyes when listening
                
                # Eye whites
                for x in [left_eye_x, right_eye_x]:
                    pygame.draw.ellipse(self.screen, (255, 255, 255), 
                                      (x - eye_width//2, eye_y - eye_height//2, 
                                       eye_width, eye_height))
                    # Eye outline
                    pygame.draw.ellipse(self.screen, (0, 0, 0), 
                                      (x - eye_width//2, eye_y - eye_height//2, 
                                       eye_width, eye_height), 2)
                
                # Iris (light blue)
                iris_color = (100, 200, 255)
                iris_radius = 16
                
                # Gentle eye movement
                self.eye_offset_x += 0.2 * (random.random() - 0.5)
                self.eye_offset_x = max(-6, min(6, self.eye_offset_x))
                self.eye_offset_y += 0.15 * (random.random() - 0.5)
                self.eye_offset_y = max(-4, min(4, self.eye_offset_y))
                
                for x in [left_eye_x, right_eye_x]:
                    # Iris
                    pygame.draw.circle(self.screen, iris_color,
                                     (int(x + self.eye_offset_x), 
                                      int(eye_y + self.eye_offset_y + 5)), iris_radius)
                    # Pupil
                    pygame.draw.circle(self.screen, (0, 0, 0),
                                     (int(x + self.eye_offset_x), 
                                      int(eye_y + self.eye_offset_y + 5)), 8)
                    # Large highlight
                    pygame.draw.circle(self.screen, (255, 255, 255),
                                     (int(x + self.eye_offset_x - 4), 
                                      int(eye_y + self.eye_offset_y + 2)), 6)
                    # Small highlight
                    pygame.draw.circle(self.screen, (255, 255, 255),
                                     (int(x + self.eye_offset_x + 5), 
                                      int(eye_y + self.eye_offset_y + 8)), 3)
                
                # Eyebrows - expressive
                brow_y = eye_y - eye_height//2 - 15
                for x in [left_eye_x, right_eye_x]:
                    brow_points = [
                        (x - 28, brow_y + 5),
                        (x - 10, brow_y),
                        (x + 10, brow_y),
                        (x + 28, brow_y + 5)
                    ]
                    pygame.draw.lines(self.screen, (80, 50, 20), False, brow_points, 4)
            
            # Nose - subtle cute nose
            nose_x = head_center[0]
            nose_y = head_center[1] + 15
            pygame.draw.ellipse(self.screen, (240, 200, 170),
                              (nose_x - 8, nose_y, 16, 12))
            
            # Mouth
            mouth_y = head_center[1] + 50
            mouth_x = head_center[0]
            
            if self.state == 'speaking':
                # Animated speaking mouth
                self.mouth_open += 1
                mouth_height = 20 + abs(12 * (self.mouth_open % 2 - 1))
                pygame.draw.ellipse(self.screen, (230, 140, 140), 
                                  (mouth_x - 30, mouth_y - 8, 60, int(mouth_height)))
                # Teeth
                pygame.draw.rect(self.screen, (255, 255, 255),
                               (mouth_x - 20, mouth_y - 5, 40, 8))
            else:
                # Friendly smile
                smile_points = [
                    (mouth_x - 35, mouth_y),
                    (mouth_x - 20, mouth_y + 8),
                    (mouth_x, mouth_y + 12),
                    (mouth_x + 20, mouth_y + 8),
                    (mouth_x + 35, mouth_y)
                ]
                pygame.draw.lines(self.screen, (220, 100, 100), False, smile_points, 5)
                # Upper lip line
                pygame.draw.arc(self.screen, (200, 90, 90),
                              (mouth_x - 35, mouth_y - 5, 70, 15), 0, 3.14, 2)
            
            # Status text
            status_y = 520
            if self.state == 'listening':
                status_text = self.small_font.render("Listening... (Speak now!)", True, (0, 255, 0))
                self.screen.blit(status_text, (280, status_y))
            elif self.state == 'processing':
                status_text = self.small_font.render("Thinking...", True, (255, 165, 0))
                self.screen.blit(status_text, (330, status_y))
            elif self.state == 'speaking':
                status_text = self.small_font.render("Speaking...", True, (0, 0, 255))
                self.screen.blit(status_text, (330, status_y))
            else:
                status_text = self.small_font.render("Ready to help! Press ESC to quit", True, (0, 128, 0))
                self.screen.blit(status_text, (250, status_y))
            
            # User text bubble
            if self.user_text:
                bubble_rect = pygame.Rect(50, 50, 300, 80)
                pygame.draw.rect(self.screen, (255, 255, 255), bubble_rect)
                pygame.draw.rect(self.screen, (0, 0, 0), bubble_rect, 2)
                user_surf = self.font.render(f"You: {self.user_text[:35]}...", True, (0, 0, 0))
                self.screen.blit(user_surf, (60, 70))
            
            # Assistant text bubble
            if self.assistant_text:
                bubble_rect = pygame.Rect(450, 50, 300, 80)
                pygame.draw.rect(self.screen, (255, 255, 255), bubble_rect)
                pygame.draw.rect(self.screen, (0, 0, 0), bubble_rect, 2)
                assist_surf = self.font.render(f"Me: {self.assistant_text[:35]}...", True, (0, 0, 0))
                self.screen.blit(assist_surf, (460, 70))
            
            # Update display
            pygame.display.flip()
            
        except Exception as e:
            print(f"⚠️ Draw error: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()

    def _init_vosk(self):
        """Initialize Vosk model"""
        model_path = os.path.join(ROOT_DIR, "models", "vosk-model-small-en-us-0.15")
        if os.path.exists(model_path):
            try:
                print("🎙️ Loading Vosk model...")
                self.vosk_model = Model(model_path)
                print("✅ Vosk loaded - using OFFLINE speech recognition")
            except Exception as e:
                print(f"⚠️ Vosk load failed: {e}")
                self.vosk_model = None

    def _configure_recognizer(self):
        """Configure speech recognizer"""
        self.recognizer.energy_threshold = 200
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.dynamic_energy_adjustment_damping = 0.15
        self.recognizer.dynamic_energy_ratio = 1.5
        self.recognizer.pause_threshold = 1.0
        self.recognizer.phrase_threshold = 0.3
        self.recognizer.non_speaking_duration = 0.5

    def _precache_common_phrases(self):
        """Pre-generate audio for common phrases"""
        common = [
            "Please look at the camera for identification.",
            "I don't see anyone in front of the camera.",
            "Hello. I don't believe we've met before.",
            "I couldn't get a clear view of your face.",
            "How may I assist you today?"
        ]
        for phrase in common:
            try:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
                gTTS(text=phrase, lang='en', slow=False).save(tmp)
                self.audio_cache[phrase] = tmp
            except Exception as e:
                if self.debug:
                    print(f"⚠️ Cache failed: {e}")

    def _print_audio_devices(self):
        try:
            print("\n📋 Available Audio Devices:")
            for i, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f" [{i}] {name}")
        except Exception as e:
            if self.debug:
                print(f"Error listing devices: {e}")

    def _test_microphone(self):
        """Test and configure microphone"""
        mic_device_index = None
        try:
            mic_list = sr.Microphone.list_microphone_names()
            for i, name in enumerate(mic_list):
                if any(kw in name.lower() for kw in ['usb', 'webcam', 'logi', 'headset', 'mic']):
                    mic_device_index = i
                    break
            
            mic = sr.Microphone(device_index=mic_device_index, sample_rate=16000, chunk_size=1024)
            with mic as source:
                print("🎤 Calibrating microphone...")
                self.recognizer.adjust_for_ambient_noise(source, duration=2.0)
                
                if self.recognizer.energy_threshold > 400:
                    self.recognizer.energy_threshold = 300
                elif self.recognizer.energy_threshold < 100:
                    self.recognizer.energy_threshold = 200
                
                print(f"✓ Mic Ready - Threshold: {int(self.recognizer.energy_threshold)}")
                self._ambient_adjusted = True
            
            self.mic_device_index = mic_device_index
            self.mic_sample_rate = 16000
            self.mic_chunk_size = 1024
            
        except Exception as e:
            print(f"⚠️ Mic test failed: {e}")
            self.mic_device_index = None
            self.mic_sample_rate = 16000
            self.mic_chunk_size = 1024
            self._ambient_adjusted = True

    def listen_blocking(self):
        """Listen for speech input"""
        try:
            print("🎤 Listening...")
            with sr.Microphone(device_index=self.mic_device_index,
                             sample_rate=16000,
                             chunk_size=1024) as source:
                
                if not hasattr(self, '_ambient_adjusted') or not self._ambient_adjusted:
                    self.recognizer.adjust_for_ambient_noise(source, duration=1.0)
                    self._ambient_adjusted = True
                
                self.state = 'listening'
                if self.screen:
                    self.draw_face()
                
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=15)
            
            self.state = 'processing'
            if self.screen:
                self.draw_face()
            
            # Try Vosk first
            if self.vosk_model:
                text = self._transcribe_vosk(audio)
                if text:
                    self.state = 'idle'
                    return text
            
            # Fallback to Google
            try:
                text = self.recognizer.recognize_google(audio, language="en-US")
                print(f"✅ Recognized: {text}")
            except sr.UnknownValueError:
                print("❓ Could not understand audio")
                text = None
            except sr.RequestError as e:
                print(f"❌ API error: {e}")
                text = None
            
            self.state = 'idle'
            return text
            
        except sr.WaitTimeoutError:
            self.state = 'idle'
            print("⏱️ Timeout - no speech detected")
            return None
        except Exception as e:
            self.state = 'idle'
            print(f"❌ Listen error: {e}")
            return None

    def _transcribe_vosk(self, audio):
        """Transcribe using Vosk"""
        try:
            wav_data = audio.get_wav_data()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(wav_data)
                tmp_path = tmp_file.name
            
            try:
                wf = wave.open(tmp_path, "rb")
                if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
                    return None
                
                rec = KaldiRecognizer(self.vosk_model, wf.getframerate())
                rec.SetWords(True)
                
                results = []
                while True:
                    data = wf.readframes(4000)
                    if len(data) == 0:
                        break
                    if rec.AcceptWaveform(data):
                        result = json.loads(rec.Result())
                        if 'text' in result:
                            results.append(result['text'])
                
                final_result = json.loads(rec.FinalResult())
                if 'text' in final_result:
                    results.append(final_result['text'])
                
                wf.close()
                text = ' '.join(results).strip()
                return text if text else None
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        except Exception as e:
            return None

    def get_ai_response(self, user_input):
        """Get AI response"""
        try:
            self.state = 'processing'
            if self.screen:
                self.draw_face()
            
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.6,
                max_tokens=100
            )
            response = completion.choices[0].message.content
            cleaned_response = clean_text_for_speech(response)
            print(f"🤖 Response: {cleaned_response}")
            return cleaned_response
        except Exception as e:
            print(f"❌ AI error: {e}")
            return "I apologize, I'm experiencing technical difficulties."

    def speak(self, text, use_cache=True):
        """Speak text using TTS"""
        self.state = 'speaking'
        self.assistant_text = text
        tmp = None
        
        try:
            clean_text = clean_text_for_speech(text) if not use_cache else text
            
            if use_cache and clean_text in self.audio_cache:
                tmp = self.audio_cache[clean_text]
            else:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
                gTTS(text=clean_text, lang='en', slow=False).save(tmp)
            
            pygame.mixer.music.load(tmp)
            pygame.mixer.music.play()
            
            start_time = time.time()
            while pygame.mixer.music.get_busy():
                if self.screen:
                    self.draw_face()
                    self.clock.tick(30)
                if time.time() - start_time > 30:
                    pygame.mixer.music.stop()
                    break
            
        except Exception as e:
            print(f"❌ Speak error: {e}")
        finally:
            self.state = 'idle'
            self.assistant_text = ""
            if tmp and not use_cache and os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except:
                    pass

    def greet_with_face(self):
        """Greet with face recognition"""
        if not FACE_RECOG_AVAILABLE:
            self.speak("The facial recognition system is unavailable.", use_cache=False)
            return
        
        self.speak("Please look at the camera.", use_cache=True)
        
        try:
            name = face_recog_bind.recognize_face_once()
            
            if not name or name == "No frame":
                self.speak("I couldn't see your face clearly.", use_cache=True)
            elif name == "No face":
                self.speak("I don't see anyone.", use_cache=True)
            elif name == "Stranger":
                self.speak("Hello. I don't believe we've met. How may I help you?", use_cache=True)
            else:
                greeting = f"Hello {name}. Welcome back. How may I help you?"
                self.speak(greeting, use_cache=False)
        except Exception as e:
            print(f"⚠️ Face recognition error: {e}")
            self.speak("Face recognition encountered an error.", use_cache=False)

    def run(self):
        """Main run loop"""
        print("="*60)
        print("OFFICE RECEPTIONIST ASSISTANT")
        print("="*60)
        print("\nCommands:")
        print(" - Say 'hello' for face recognition")
        print(" - Say 'exit' to quit")
        print(" - Press ESC to quit\n")
        
        self.speak("Office receptionist ready. How may I help you?", use_cache=False)
        
        running = True
        while running:
            # Handle GUI events
            if self.screen:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                
                self.draw_face()
                self.clock.tick(30)
            
            # Listen for input
            user_input = self.listen_blocking()
            if user_input is None:
                continue
            
            text = user_input.lower().strip()
            self.user_text = user_input
            
            # Exit
            if any(word in text for word in ["exit", "quit", "goodbye"]):
                self.speak("Goodbye. Have a pleasant day.", use_cache=False)
                running = False
                break
            
            # Face recognition
            elif "hello" in text or "identify" in text:
                self.greet_with_face()
                
                remaining = None
                if "hello" in text:
                    parts = text.split("hello", 1)
                    if len(parts) > 1:
                        remaining = parts[1].strip()
                
                if remaining and len(remaining) > 3:
                    reply = self.get_ai_response(remaining)
                    self.speak(reply, use_cache=False)
            
            # Normal conversation
            else:
                reply = self.get_ai_response(text)
                self.speak(reply, use_cache=False)
            
            self.user_text = ""
            self.state = 'idle'
        
        # Cleanup
        print("\n🧹 Cleaning up...")
        self.executor.shutdown(wait=False)
        for tmp in self.audio_cache.values():
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except:
                    pass
        
        if self.screen:
            pygame.quit()
        print("✅ Shutdown complete")

def main():
    print("Starting Office Receptionist Assistant...\n")
    debug = "--debug" in sys.argv
    if debug:
        print("Debug mode enabled\n")
    
    try:
        assistant = JetsonVoiceAssistant(debug=debug)
        assistant.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nGoodbye!")

if __name__ == "__main__":
    main()
