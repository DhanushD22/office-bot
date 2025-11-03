#!/usr/bin/env python3
"""
Voice Assistant optimized for NVIDIA Jetson Nano - Office Receptionist Persona
OPTIMIZED VERSION - Fast and Reliable
"""

import os
import sys
import subprocess
import time
import ctypes
import tempfile
import warnings
import pygame
from concurrent.futures import ThreadPoolExecutor
import atexit
import re

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
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import speech_recognition as sr
from gtts import gTTS
from openai import OpenAI
warnings.filterwarnings("ignore")

# Try to import Vosk for better offline accuracy
VOSK_AVAILABLE = False
try:
    from vosk import Model, KaldiRecognizer
    import json
    import wave
    VOSK_AVAILABLE = True
    print("✅ Vosk available for offline speech recognition")
except ImportError:
    print("ℹ️ Vosk not installed. Using Google STT (online)")
    print("   To install: pip3 install vosk")

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

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="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",
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
- Always be respectful and courteous

Example responses:
Bad: "**Hello!** I can help you with:\n- Scheduling\n- Directions\n- Information"
Good: "Hello, I can help you with scheduling, directions, or office information."

Bad: "The meeting room is on the *2nd floor* (near the **elevator**)"
Good: "The meeting room is on the second floor near the elevator."

Remember: You are speaking out loud, so write exactly how you would speak."""

        # Speech recognizer setup - SIMPLIFIED AND OPTIMIZED
        self.recognizer = sr.Recognizer()
        self._configure_recognizer()

        # Audio setup
        self._init_audio()
        self._setup_jetson_audio()
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Initialize Vosk if available
        self.vosk_model = None
        if VOSK_AVAILABLE:
            self._init_vosk()
        
        # Pre-cache audio files for common responses
        self.audio_cache = {}
        self._precache_common_phrases()

        print("\n✅ Office Receptionist Assistant Ready!\n")

    def _init_vosk(self):
        """Initialize Vosk model for offline recognition"""
        model_path = os.path.join(ROOT_DIR, "models", "vosk-model-small-en-us-0.15")
        
        if os.path.exists(model_path):
            try:
                print("🎙️ Loading Vosk model...")
                self.vosk_model = Model(model_path)
                print("✅ Vosk loaded - using OFFLINE speech recognition")
            except Exception as e:
                print(f"⚠️ Vosk load failed: {e}")
                self.vosk_model = None
        else:
            print(f"ℹ️ Vosk model not found at: {model_path}")
            print("   Download from: https://alphacephei.com/vosk/models")
            print("   Using Google STT instead")

    def _configure_recognizer(self):
        """OPTIMIZED: Better accuracy settings"""
        # Fixed threshold for consistency
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True  # Re-enabled for better accuracy
        self.recognizer.dynamic_energy_adjustment_damping = 0.15
        self.recognizer.dynamic_energy_ratio = 1.5
        
        # Better pause detection for complete sentences
        self.recognizer.pause_threshold = 1.0  # Wait 1 second before ending
        self.recognizer.phrase_threshold = 0.3
        self.recognizer.non_speaking_duration = 0.5

    def _init_audio(self):
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=2048)
            print("🎧 Audio initialized")
        except Exception as e:
            print(f"⚠️ Audio warning: {e}")
            try:
                pygame.mixer.init()
            except:
                pass

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
            except:
                pass

    def _setup_jetson_audio(self):
        try:
            if self.debug:
                self._print_audio_devices()
            self._test_microphone()
        except Exception as e:
            print(f"⚠️ Audio setup warning: {e}")

    def _print_audio_devices(self):
        try:
            print("\n📋 Available Audio Devices:")
            for i, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"  [{i}] {name}")
        except Exception as e:
            if self.debug:
                print(f"Error listing devices: {e}")

    def _test_microphone(self):
        """OPTIMIZED: Quick and simple mic test"""
        mic_device_index = None
        
        try:
            mic_list = sr.Microphone.list_microphone_names()
            
            # Find USB or external microphone
            for i, name in enumerate(mic_list):
                if any(kw in name.lower() for kw in ['usb', 'webcam', 'logi', 'headset', 'mic']):
                    mic_device_index = i
                    break

            print("🔧 Testing microphone...")
            mic = sr.Microphone(device_index=mic_device_index, sample_rate=16000, chunk_size=1024)
            
            with mic as source:
                print("   Calibrating... speak in 1 second...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1.0)
                
                # Override if too high
                if self.recognizer.energy_threshold > 400:
                    self.recognizer.energy_threshold = 300
                    print(f"   Adjusted threshold to 300")
                
                energy = int(self.recognizer.energy_threshold)
                print(f"✓ Mic Ready: {mic_list[mic_device_index] if mic_device_index else 'default'}")
                print(f"✓ Threshold: {energy}")
                
                # Mark as adjusted so we don't do it again
                self._ambient_adjusted = False  # Will do it once on first listen()
            
            self.mic_device_index = mic_device_index
            self.mic_sample_rate = 16000
            self.mic_chunk_size = 1024
            
        except Exception as e:
            print(f"⚠️ Mic test failed: {e}")
            print("⚠️ Using default microphone")
            self.mic_device_index = None
            self.mic_sample_rate = 16000
            self.mic_chunk_size = 1024

    def listen(self):
        """OPTIMIZED: Immediate listening with better accuracy"""
        try:
            print("🎤 Listening... speak now!")
            
            with sr.Microphone(device_index=self.mic_device_index,
                               sample_rate=16000,
                               chunk_size=1024) as source:
                
                # FIXED: Quick ambient adjustment only on first call
                if not hasattr(self, '_ambient_adjusted'):
                    print("   Calibrating for ambient noise...")
                    self.recognizer.adjust_for_ambient_noise(source, duration=1.0)
                    self._ambient_adjusted = True
                
                # Start listening IMMEDIATELY - no delays
                audio = self.recognizer.listen(
                    source, 
                    timeout=10,  # Wait longer for user to start speaking
                    phrase_time_limit=15  # Allow longer phrases
                )
            
            print("   🔄 Processing speech...")
            
            # Try Vosk first (offline, fast, accurate)
            if self.vosk_model:
                text = self._transcribe_vosk(audio)
                if text:
                    print(f"✅ You said: {text}")
                    return text
            
            # Fallback to Google STT
            try:
                # Try with US English for better accuracy
                text = self.recognizer.recognize_google(
                    audio, 
                    language="en-US",
                    show_all=False
                )
            except:
                # Fallback to Indian English
                text = self.recognizer.recognize_google(audio, language="en-IN")
            
            if text:
                print(f"✅ You said: {text}")
                return text
            else:
                return None
                
        except sr.WaitTimeoutError:
            print("⏱️ No speech detected - please speak after the prompt")
            return None
        except sr.UnknownValueError:
            print("❓ Could not understand - please speak clearly")
            return None
        except sr.RequestError as e:
            print(f"❌ Recognition error: {e}")
            return None
        except Exception as e:
            print(f"❌ Error: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return None

    def _transcribe_vosk(self, audio):
        """Transcribe using Vosk (offline, accurate)"""
        try:
            # Convert to WAV format
            wav_data = audio.get_wav_data()
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(wav_data)
                tmp_path = tmp_file.name
            
            try:
                # Open WAV file
                wf = wave.open(tmp_path, "rb")
                
                # Check format
                if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() not in [8000, 16000, 32000, 48000]:
                    print("   ⚠️ Audio format not optimal for Vosk")
                    return None
                
                # Create recognizer
                rec = KaldiRecognizer(self.vosk_model, wf.getframerate())
                rec.SetWords(True)
                
                # Process audio
                results = []
                while True:
                    data = wf.readframes(4000)
                    if len(data) == 0:
                        break
                    if rec.AcceptWaveform(data):
                        result = json.loads(rec.Result())
                        if 'text' in result:
                            results.append(result['text'])
                
                # Get final result
                final_result = json.loads(rec.FinalResult())
                if 'text' in final_result:
                    results.append(final_result['text'])
                
                wf.close()
                
                # Combine results
                text = ' '.join(results).strip()
                return text if text else None
                
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                    
        except Exception as e:
            if self.debug:
                print(f"   Vosk error: {e}")
            return None

    def get_ai_response(self, user_input):
        """Query LLM with receptionist persona"""
        try:
            if self.debug:
                print("🧠 Processing...")
            
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
            
            print(f"🤖 Assistant: {cleaned_response}")
            return cleaned_response
            
        except Exception as e:
            print(f"❌ AI error: {e}")
            return "I apologize, I'm experiencing technical difficulties. Please try again."

    def speak(self, text, use_cache=True):
        """Speak text using gTTS + pygame"""
        tmp = None
        try:
            clean_text = clean_text_for_speech(text) if not use_cache else text
            
            # Check cache
            if use_cache and clean_text in self.audio_cache:
                tmp = self.audio_cache[clean_text]
            else:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
                gTTS(text=clean_text, lang='en', slow=False).save(tmp)
            
            # Play audio
            pygame.mixer.music.load(tmp)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
                
        except Exception as e:
            if self.debug:
                print(f"⚠️ Speak error: {e}")
            print(f"🗣️ {text}")
        finally:
            if tmp and not use_cache and os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except:
                    pass

    def greet_with_face(self):
        """Fast personalized greeting"""
        if not FACE_RECOG_AVAILABLE:
            self.speak("Good day. The facial recognition system is currently unavailable.", use_cache=False)
            return

        self.speak("Please look at the camera for identification.", use_cache=True)
        
        try:
            start_time = time.time()
            name = face_recog_bind.recognize_face_once()
            elapsed = time.time() - start_time
            
            if self.debug:
                print(f"⏱️ Recognition: {elapsed:.2f}s")

            if not name or name == "No frame":
                self.speak("I couldn't get a clear view of your face.", use_cache=True)
            elif name == "No face":
                self.speak("I don't see anyone in front of the camera.", use_cache=True)
            elif name == "Stranger":
                self.speak("Hello. I don't believe we've met before. How may I assist you today?", use_cache=True)
            elif name == "No database images":
                self.speak("The facial recognition database is empty. Please contact the administrator.", use_cache=False)
            elif name == "Camera error":
                self.speak("I cannot access the camera. Please check the connection.", use_cache=False)
            elif name.startswith("Error:"):
                self.speak("I'm experiencing technical difficulties. Please try again.", use_cache=False)
            else:
                greeting = f"Good day {name}. Welcome back. How may I assist you today?"
                self.speak(greeting, use_cache=False)

        except Exception as e:
            print(f"⚠️ Face recognition error: {e}")
            self.speak("I apologize. The facial recognition system encountered an error.", use_cache=False)

    def run(self):
        print("="*60)
        print("🏢  OFFICE RECEPTIONIST ASSISTANT")
        print("="*60)
        print("\n💡 Commands:")
        print("   • 'habibi [request]' - Face recognition + request")
        print("   • 'identify me' - Just face recognition")
        print("   • 'exit' - Quit assistant")
        print("   • 'reload faces' - Update database")
        print("\n📢 Examples:")
        print("   • 'habibi tell me a joke'")
        print("   • 'habibi what time is it'")
        print("   • 'what's the weather like'\n")

        self.speak("Good day. Office receptionist assistant is ready. How may I help you?", use_cache=False)

        while True:
            user_input = self.listen()
            if not user_input:
                continue
            
            text = user_input.lower().strip()

            # Exit command
            if any(word in text for word in ["exit", "quit", "goodbye", "bye"]):
                self.speak("Goodbye. Have a pleasant day.", use_cache=False)
                break
            
            # Face recognition with optional request
            elif "habibi" in text or "identify" in text or "recognize me" in text:
                # Do face recognition
                self.greet_with_face()
                
                # Extract request after keyword
                remaining_request = None
                
                if "habibi" in text:
                    parts = text.split("habibi", 1)
                    if len(parts) > 1:
                        remaining_request = parts[1].strip()
                elif "identify me" in text:
                    parts = text.split("identify me", 1)
                    if len(parts) > 1:
                        remaining_request = parts[1].strip()
                elif "recognize me" in text:
                    parts = text.split("recognize me", 1)
                    if len(parts) > 1:
                        remaining_request = parts[1].strip()
                
                # Process additional request if exists
                if remaining_request and len(remaining_request) > 3:
                    print(f"📝 Processing: {remaining_request}")
                    reply = self.get_ai_response(remaining_request)
                    self.speak(reply, use_cache=False)
            
            # Reload database
            elif "reload" in text and "face" in text:
                if FACE_RECOG_AVAILABLE:
                    face_recog_bind.reload_database()
                    db_size = face_recog_bind.get_database_size()
                    self.speak(f"Face database updated. {db_size} profiles available.", use_cache=False)
                else:
                    self.speak("Face recognition is not available.", use_cache=False)
            
            # Normal conversation
            else:
                reply = self.get_ai_response(text)
                self.speak(reply, use_cache=False)
        
        # Cleanup
        self.executor.shutdown(wait=False)
        for tmp in self.audio_cache.values():
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except:
                    pass

def main():
    print("🚀 Starting Office Receptionist Assistant...\n")
    debug = "--debug" in sys.argv
    assistant = JetsonVoiceAssistant(debug=debug)
    assistant.run()

if __name__ == "__main__":
    main()
