"""
Free AI Services for Grievance Portal
Uses free alternatives to OpenAI for image analysis and audio transcription
"""

import os
import base64
import io
import tempfile
from PIL import Image
import json
from typing import Dict, Optional, Tuple
import logging
import requests
import cv2
import numpy as np
import easyocr
import speech_recognition as sr
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FreeAIServices:
    def __init__(self):
        """Initialize free AI services"""
        self.ocr_reader = None
        self.image_captioning_model = None
        self.image_processor = None
        self.speech_recognizer = None
        self.text_generator = None
        self.initialize_services()
    
    def initialize_services(self):
        """Initialize all free AI services"""
        try:
            # Initialize OCR for text extraction from images
            logger.info("Initializing EasyOCR...")
            self.ocr_reader = easyocr.Reader(['en'])
            
            # Initialize image captioning model (BLIP)
            logger.info("Initializing BLIP image captioning model...")
            self.image_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.image_captioning_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            
            # Initialize speech recognition
            self.speech_recognizer = sr.Recognizer()
            
            # Initialize text generation pipeline
            logger.info("Initializing text generation model...")
            self.text_generator = pipeline(
                "text-generation", 
                model="microsoft/DialoGPT-medium",
                max_length=200,
                do_sample=True,
                temperature=0.7
            )
            
            logger.info("Free AI services initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize free AI services: {e}")
            return False
    
    def analyze_image_for_grievance(self, image_data: bytes, image_format: str = "JPEG") -> Dict:
        """
        Analyze uploaded image using free AI services
        Combines OCR text extraction + image captioning + rule-based analysis
        """
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to OpenCV format for OCR
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            results = {
                "ocr_text": "",
                "image_caption": "",
                "detected_issues": [],
                "confidence": 0.0
            }
            
            # 1. Extract text using OCR
            try:
                ocr_results = self.ocr_reader.readtext(cv_image)
                ocr_text = " ".join([text[1] for text in ocr_results if text[2] > 0.5])
                results["ocr_text"] = ocr_text
                logger.info(f"OCR extracted text: {ocr_text[:100]}...")
            except Exception as e:
                logger.warning(f"OCR failed: {e}")
            
            # 2. Generate image caption
            try:
                inputs = self.image_processor(image, return_tensors="pt")
                out = self.image_captioning_model.generate(**inputs, max_length=50)
                caption = self.image_processor.decode(out[0], skip_special_tokens=True)
                results["image_caption"] = caption
                logger.info(f"Image caption: {caption}")
            except Exception as e:
                logger.warning(f"Image captioning failed: {e}")
                results["image_caption"] = "Unable to generate caption"
            
            # 3. Detect common issues based on keywords and image analysis
            detected_issues = self.detect_issues_from_image_content(
                results["ocr_text"], 
                results["image_caption"]
            )
            results["detected_issues"] = detected_issues
            
            # 4. Generate grievance text
            grievance_text = self.generate_grievance_from_analysis(results)
            
            # 5. Calculate confidence based on available information
            confidence = self.calculate_confidence(results)
            
            return {
                "success": True,
                "generated_text": grievance_text["description"],
                "subject": grievance_text["subject"],
                "confidence": confidence,
                "analysis_details": results,
                "analysis_method": "Free AI (OCR + BLIP + Rule-based)"
            }
            
        except Exception as e:
            logger.error(f"Free image analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "generated_text": "",
                "confidence": 0.0
            }
    
    def detect_issues_from_image_content(self, ocr_text: str, caption: str) -> list:
        """Detect common grievance issues from image content"""
        issues = []
        combined_text = f"{ocr_text} {caption}".lower()
        
        # Infrastructure issues
        infrastructure_keywords = [
            "road", "pothole", "crack", "damage", "broken", "street", "highway",
            "bridge", "construction", "repair", "maintenance"
        ]
        
        # Water issues
        water_keywords = [
            "water", "leak", "pipe", "drainage", "flood", "sewage", "overflow",
            "supply", "tank", "pump", "well"
        ]
        
        # Electricity issues
        electricity_keywords = [
            "power", "electricity", "wire", "pole", "transformer", "outage",
            "cable", "electric", "current", "voltage"
        ]
        
        # Public services
        public_keywords = [
            "garbage", "waste", "trash", "clean", "sanitation", "hospital",
            "school", "office", "government", "public"
        ]
        
        # Environmental issues
        environment_keywords = [
            "pollution", "dirty", "contamination", "environment", "tree",
            "park", "green", "nature", "air", "noise"
        ]
        
        if any(keyword in combined_text for keyword in infrastructure_keywords):
            issues.append("infrastructure")
        if any(keyword in combined_text for keyword in water_keywords):
            issues.append("water_supply")
        if any(keyword in combined_text for keyword in electricity_keywords):
            issues.append("electricity")
        if any(keyword in combined_text for keyword in public_keywords):
            issues.append("public_services")
        if any(keyword in combined_text for keyword in environment_keywords):
            issues.append("environment")
        
        return issues if issues else ["general"]
    
    def generate_grievance_from_analysis(self, analysis_results: Dict) -> Dict:
        """Generate formal grievance text from analysis results"""
        ocr_text = analysis_results.get("ocr_text", "")
        caption = analysis_results.get("image_caption", "")
        issues = analysis_results.get("detected_issues", [])
        
        # Generate subject based on detected issues
        if "infrastructure" in issues:
            subject = "Request for Infrastructure Maintenance and Repair"
        elif "water_supply" in issues:
            subject = "Water Supply Issue - Request for Resolution"
        elif "electricity" in issues:
            subject = "Electrical Infrastructure Problem - Urgent Attention Required"
        elif "public_services" in issues:
            subject = "Public Service Issue - Request for Improvement"
        elif "environment" in issues:
            subject = "Environmental Concern - Request for Action"
        else:
            # Try to generate subject based on caption
            if caption and caption != "Unable to generate caption":
                # Extract key words for subject
                caption_lower = caption.lower()
                if "road" in caption_lower or "street" in caption_lower:
                    subject = "Road Maintenance Issue - Request for Repair"
                elif "building" in caption_lower or "structure" in caption_lower:
                    subject = "Building/Structure Issue - Request for Inspection"
                elif "water" in caption_lower or "drainage" in caption_lower:
                    subject = "Water/Drainage Issue - Request for Resolution"
                elif "garbage" in caption_lower or "waste" in caption_lower:
                    subject = "Waste Management Issue - Request for Action"
                elif "light" in caption_lower or "electricity" in caption_lower:
                    subject = "Lighting/Electrical Issue - Request for Repair"
                else:
                    subject = "Public Infrastructure Issue - Request for Attention"
            else:
                subject = "Public Infrastructure Issue - Request for Attention"
        
        # Generate description
        description_parts = []
        
        if caption and caption != "Unable to generate caption":
            # Remove the "Based on the submitted image" prefix for cleaner suggestions
            description_parts.append(f"{caption.capitalize()}.")
        
        if ocr_text:
            description_parts.append(f"Text visible in the image: {ocr_text}")
        
        if issues:
            issue_descriptions = {
                "infrastructure": "This appears to be related to road or infrastructure maintenance requirements.",
                "water_supply": "This issue is related to water supply or drainage problems.",
                "electricity": "This concerns electrical infrastructure or power supply issues.",
                "public_services": "This relates to public services or facilities that require attention.",
                "environment": "This is an environmental concern that needs to be addressed."
            }
            
            for issue in issues:
                if issue in issue_descriptions:
                    description_parts.append(issue_descriptions[issue])
        
        description_parts.append("I request the concerned authorities to take appropriate action to resolve this matter at the earliest.")
        
        description = " ".join(description_parts)
        
        return {
            "subject": subject,
            "description": description
        }
    
    def calculate_confidence(self, analysis_results: Dict) -> float:
        """Calculate confidence score based on available information"""
        confidence = 0.3  # Base confidence
        
        if analysis_results.get("ocr_text"):
            confidence += 0.3  # OCR text found
        
        if analysis_results.get("image_caption") and "unable" not in analysis_results["image_caption"].lower():
            confidence += 0.3  # Good caption generated
        
        if analysis_results.get("detected_issues"):
            confidence += 0.2  # Issues detected
        
        return min(confidence, 0.95)  # Cap at 95%
    
    def transcribe_audio_to_text(self, audio_data: bytes, audio_format: str = "wav") -> Dict:
        """
        Transcribe audio using free speech recognition
        Uses Google Speech Recognition (free tier)
        
        Note: This function works best with PCM WAV files (16-bit, mono, 16-44kHz)
        """
        try:
            logger.info(f"Starting audio transcription for format: {audio_format}")
            
            # Create temporary file for audio
            with tempfile.NamedTemporaryFile(suffix=f'.{audio_format}', delete=False) as temp_audio:
                temp_audio.write(audio_data)
                temp_audio_path = temp_audio.name
            
            try:
                logger.info(f"Processing audio file: {temp_audio_path}")
                
                # For WAV files, try direct processing first
                if audio_format.lower() == 'wav':
                    logger.info("Attempting direct WAV processing...")
                    try:
                        with sr.AudioFile(temp_audio_path) as source:
                            # Adjust for ambient noise
                            self.speech_recognizer.adjust_for_ambient_noise(source, duration=0.5)
                            audio = self.speech_recognizer.record(source)
                        
                        # Try Google Speech Recognition (free tier)
                        logger.info("Calling Google Speech Recognition API...")
                        transcribed_text = self.speech_recognizer.recognize_google(audio, language='en-US')
                        
                        logger.info(f"Direct WAV transcription successful: {transcribed_text[:50]}...")
                        
                        return {
                            "success": True,
                            "transcribed_text": transcribed_text,
                            "confidence": 0.8,
                            "transcription_method": "Google Speech Recognition (Direct WAV)",
                            "word_count": len(transcribed_text.split())
                        }
                        
                    except Exception as direct_error:
                        logger.warning(f"Direct WAV processing failed: {direct_error}")
                        # Continue to conversion attempt
                
                # Try conversion using pydub (with or without ffmpeg)
                try:
                    from pydub import AudioSegment
                    from pydub.utils import which
                    
                    # Check if ffmpeg is available
                    has_ffmpeg = which("ffmpeg") is not None
                    logger.info(f"ffmpeg available: {has_ffmpeg}")
                    
                    if audio_format.lower() == 'wav' and not has_ffmpeg:
                        # For WAV files without ffmpeg, try basic conversion
                        logger.info("Attempting basic WAV conversion without ffmpeg...")
                        audio = AudioSegment.from_wav(temp_audio_path)
                    else:
                        # For other formats or when ffmpeg is available
                        logger.info(f"Loading audio file in {audio_format} format...")
                        audio = AudioSegment.from_file(temp_audio_path, format=audio_format)
                    
                    logger.info(f"Original audio: {audio.frame_rate}Hz, {audio.channels} channels, {audio.sample_width * 8}-bit")
                    
                    # Convert to speech recognition friendly format
                    # Use reasonable sample rate (16kHz is good for speech)
                    target_rate = 16000 if audio.frame_rate > 16000 else audio.frame_rate
                    audio = audio.set_frame_rate(target_rate)
                    audio = audio.set_channels(1)  # Mono
                    audio = audio.set_sample_width(2)  # 16-bit
                    
                    # Export as PCM WAV
                    wav_path = temp_audio_path.replace(f'.{audio_format}', '_converted.wav')
                    
                    if has_ffmpeg:
                        audio.export(wav_path, format="wav", parameters=["-acodec", "pcm_s16le"])
                    else:
                        # Basic export without codec specification
                        audio.export(wav_path, format="wav")
                    
                    logger.info(f"Audio converted successfully to: {wav_path}")
                    
                    # Process the converted file
                    with sr.AudioFile(wav_path) as source:
                        self.speech_recognizer.adjust_for_ambient_noise(source, duration=0.5)
                        audio_data = self.speech_recognizer.record(source)
                    
                    # Try Google Speech Recognition
                    logger.info("Calling Google Speech Recognition API...")
                    transcribed_text = self.speech_recognizer.recognize_google(audio_data, language='en-US')
                    
                    logger.info(f"Converted audio transcription successful: {transcribed_text[:50]}...")
                    
                    # Clean up converted file
                    if os.path.exists(wav_path):
                        os.unlink(wav_path)
                    
                    return {
                        "success": True,
                        "transcribed_text": transcribed_text,
                        "confidence": 0.8,
                        "transcription_method": "Google Speech Recognition (Converted)",
                        "word_count": len(transcribed_text.split())
                    }
                    
                except ImportError:
                    logger.error("pydub not available")
                    return {
                        "success": False,
                        "error": "Audio conversion not possible. Please upload a PCM WAV file (16-bit, mono, 16-44kHz) or install pydub.",
                        "transcribed_text": "",
                        "confidence": 0.0
                    }
                except Exception as conversion_error:
                    logger.error(f"Audio conversion failed: {conversion_error}")
                    return {
                        "success": False,
                        "error": f"Audio conversion failed: {conversion_error}. Please try uploading a standard PCM WAV file.",
                        "transcribed_text": "",
                        "confidence": 0.0
                    }
                
            except sr.UnknownValueError:
                logger.warning("Could not understand audio content")
                return {
                    "success": False,
                    "error": "Could not understand audio. Please speak clearly and try again.",
                    "transcribed_text": "",
                    "confidence": 0.0
                }
            except sr.RequestError as e:
                logger.error(f"Speech recognition service error: {e}")
                return {
                    "success": False,
                    "error": f"Speech recognition service error: {e}. Please check your internet connection.",
                    "transcribed_text": "",
                    "confidence": 0.0
                }
            except Exception as e:
                logger.error(f"Audio processing error: {e}")
                return {
                    "success": False,
                    "error": f"Audio processing failed: {e}",
                    "transcribed_text": "",
                    "confidence": 0.0
                }
            finally:
                # Clean up temporary files
                try:
                    if os.path.exists(temp_audio_path):
                        os.unlink(temp_audio_path)
                except Exception as cleanup_error:
                    logger.warning(f"Cleanup error: {cleanup_error}")
            
        except Exception as e:
            logger.error(f"Free audio transcription failed: {e}")
            return {
                "success": False,
                "error": f"Audio transcription failed: {str(e)}",
                "transcribed_text": "",
                "confidence": 0.0
            }
    
    def enhance_transcribed_text_for_grievance(self, transcribed_text: str) -> Dict:
        """
        Enhance transcribed text using rule-based formatting
        Since we're using free services, we'll use template-based enhancement
        """
        try:
            # Simple rule-based enhancement
            text = transcribed_text.strip()
            
            # Capitalize first letter
            if text:
                text = text[0].upper() + text[1:]
            
            # Add proper punctuation if missing
            if text and not text.endswith(('.', '!', '?')):
                text += '.'
            
            # Generate subject from first few words
            words = text.split()
            if len(words) >= 3:
                subject = " ".join(words[:5]) + ("..." if len(words) > 5 else "")
            else:
                subject = "Voice-recorded grievance"
            
            # Format as formal grievance
            enhanced_text = f"I would like to bring to your attention the following issue: {text} I request the concerned authorities to take appropriate action to resolve this matter promptly."
            
            return {
                "success": True,
                "enhanced_text": enhanced_text,
                "subject": subject,
                "enhancement_method": "Rule-based formatting"
            }
            
        except Exception as e:
            logger.error(f"Text enhancement failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "enhanced_text": transcribed_text
            }
    
    def process_image_file(self, file_content: bytes, filename: str) -> Dict:
        """Process uploaded image file using free services"""
        try:
            # Validate file type
            allowed_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
            file_ext = os.path.splitext(filename)[1].lower()
            
            if file_ext not in allowed_extensions:
                return {
                    "success": False,
                    "error": f"Unsupported file type: {file_ext}. Allowed: {', '.join(allowed_extensions)}"
                }
            
            # Validate and process image
            try:
                image = Image.open(io.BytesIO(file_content))
                
                # Convert to RGB if necessary
                if image.mode in ('RGBA', 'LA', 'P'):
                    image = image.convert('RGB')
                
                # Resize if too large (for processing efficiency)
                max_size = 1024
                if image.width > max_size or image.height > max_size:
                    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
                # Convert back to bytes
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='JPEG', quality=85)
                processed_image_data = img_byte_arr.getvalue()
                
                # Analyze the image using free services
                analysis_result = self.analyze_image_for_grievance(processed_image_data, "JPEG")
                
                # Add file info to result
                analysis_result.update({
                    "original_filename": filename,
                    "file_size": len(file_content),
                    "processed_size": len(processed_image_data),
                    "image_dimensions": f"{image.width}x{image.height}"
                })
                
                return analysis_result
                
            except Exception as img_error:
                return {
                    "success": False,
                    "error": f"Invalid image file: {str(img_error)}"
                }
            
        except Exception as e:
            logger.error(f"Free image processing failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def process_audio_file(self, file_content: bytes, filename: str) -> Dict:
        """Process uploaded audio file using free services"""
        try:
            # Validate file type
            allowed_extensions = ['.wav', '.mp3', '.m4a', '.ogg', '.flac']
            file_ext = os.path.splitext(filename)[1].lower()
            
            if file_ext not in allowed_extensions:
                return {
                    "success": False,
                    "error": f"Unsupported audio format: {file_ext}. Allowed: {', '.join(allowed_extensions)}"
                }
            
            # Check file size (reasonable limit for free services)
            max_size = 10 * 1024 * 1024  # 10MB
            if len(file_content) > max_size:
                return {
                    "success": False,
                    "error": f"Audio file too large. Maximum size is 10MB, got {len(file_content) / (1024*1024):.1f}MB"
                }
            
            # Transcribe the audio
            transcription_result = self.transcribe_audio_to_text(file_content, file_ext[1:])
            
            if transcription_result["success"]:
                # Enhance the transcribed text
                enhancement_result = self.enhance_transcribed_text_for_grievance(
                    transcription_result["transcribed_text"]
                )
                
                # Combine results
                final_result = {
                    **transcription_result,
                    "original_filename": filename,
                    "file_size": len(file_content),
                    "enhanced_text": enhancement_result.get("enhanced_text", transcription_result["transcribed_text"]),
                    "subject": enhancement_result.get("subject", "Voice-recorded grievance"),
                    "enhancement_success": enhancement_result["success"]
                }
                
                return final_result
            else:
                return transcription_result
            
        except Exception as e:
            logger.error(f"Free audio processing failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

# Initialize free AI services
try:
    free_ai_services = FreeAIServices()
    logger.info("Free AI services initialized and ready")
except Exception as e:
    logger.error(f"Failed to initialize free AI services: {e}")
    free_ai_services = None