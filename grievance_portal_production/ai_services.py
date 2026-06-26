"""
AI Services for Grievance Portal
Handles image analysis and audio transcription using OpenAI APIs
"""

import os
import base64
import io
import tempfile
from PIL import Image
from openai import OpenAI
import json
from typing import Dict, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIServices:
    def __init__(self):
        """Initialize OpenAI client with API key from environment"""
        self.client = None
        self.initialize_openai()
    
    def initialize_openai(self):
        """Initialize OpenAI client"""
        try:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OPENAI_API_KEY not found in environment variables")
                return False
            
            self.client = OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            return False
    
    def analyze_image_for_grievance(self, image_data: bytes, image_format: str = "JPEG") -> Dict:
        """
        Analyze uploaded image to generate grievance text using OpenAI Vision API
        
        Args:
            image_data: Binary image data
            image_format: Image format (JPEG, PNG, etc.)
            
        Returns:
            Dict containing generated text, confidence, and analysis details
        """
        try:
            if not self.client:
                return {
                    "success": False,
                    "error": "OpenAI client not initialized",
                    "generated_text": "",
                    "confidence": 0.0
                }
            
            # Encode image to base64
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Create the prompt for grievance analysis
            grievance_prompt = """
            Analyze this image and generate a detailed grievance/complaint text that a citizen might file with the Tamil Nadu government. 

            Instructions:
            1. Identify the main issue visible in the image (infrastructure problems, environmental issues, public services, etc.)
            2. Describe the specific problem clearly and objectively
            3. Mention the location type if identifiable (road, building, public facility, etc.)
            4. Include relevant details that would help government officials understand the issue
            5. Write in a formal, respectful tone suitable for official grievance submission
            6. Keep the text between 100-300 words
            7. Focus on actionable issues that government departments can address

            Generate a grievance text that starts with a clear subject line, followed by detailed description.
            Format: 
            Subject: [Brief subject line]
            Description: [Detailed description of the issue]
            """
            
            # Call OpenAI Vision API
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Use the vision-capable model
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": grievance_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{image_format.lower()};base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            generated_text = response.choices[0].message.content.strip()
            
            # Parse subject and description
            subject = ""
            description = ""
            
            lines = generated_text.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('Subject:'):
                    subject = line.replace('Subject:', '').strip()
                elif line.startswith('Description:'):
                    description = '\n'.join(lines[i:]).replace('Description:', '').strip()
                    break
            
            # If parsing fails, use the entire text as description
            if not description:
                description = generated_text
                subject = "Issue identified from uploaded image"
            
            # Calculate confidence based on response length and content quality
            confidence = min(0.9, max(0.3, len(description) / 200))
            
            logger.info(f"Image analysis completed successfully. Generated {len(description)} characters")
            
            return {
                "success": True,
                "generated_text": description,
                "subject": subject,
                "confidence": confidence,
                "word_count": len(description.split()),
                "analysis_method": "OpenAI GPT-4 Vision"
            }
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "generated_text": "",
                "confidence": 0.0
            }
    
    def transcribe_audio_to_text(self, audio_data: bytes, audio_format: str = "mp3") -> Dict:
        """
        Transcribe audio to text using OpenAI Whisper API
        
        Args:
            audio_data: Binary audio data
            audio_format: Audio format (mp3, wav, etc.)
            
        Returns:
            Dict containing transcribed text and analysis details
        """
        try:
            if not self.client:
                return {
                    "success": False,
                    "error": "OpenAI client not initialized",
                    "transcribed_text": "",
                    "confidence": 0.0
                }
            
            # Create a temporary file for the audio
            with tempfile.NamedTemporaryFile(suffix=f'.{audio_format}', delete=False) as temp_audio:
                temp_audio.write(audio_data)
                temp_audio_path = temp_audio.name
            
            try:
                # Transcribe using Whisper
                with open(temp_audio_path, 'rb') as audio_file:
                    transcript = self.client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language="en"  # Can be set to "ta" for Tamil if needed
                    )
                
                transcribed_text = transcript.text.strip()
                
                # Calculate confidence based on transcription length
                confidence = min(0.95, max(0.5, len(transcribed_text) / 100))
                
                logger.info(f"Audio transcription completed successfully. Transcribed {len(transcribed_text)} characters")
                
                return {
                    "success": True,
                    "transcribed_text": transcribed_text,
                    "confidence": confidence,
                    "word_count": len(transcribed_text.split()),
                    "transcription_method": "OpenAI Whisper"
                }
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
            
        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "transcribed_text": "",
                "confidence": 0.0
            }
    
    def enhance_transcribed_text_for_grievance(self, transcribed_text: str) -> Dict:
        """
        Enhance raw transcribed text to create a proper grievance format
        
        Args:
            transcribed_text: Raw transcribed text from audio
            
        Returns:
            Dict containing enhanced grievance text
        """
        try:
            if not self.client:
                return {
                    "success": False,
                    "error": "OpenAI client not initialized",
                    "enhanced_text": transcribed_text
                }
            
            enhancement_prompt = f"""
            Convert the following raw transcribed audio into a formal grievance/complaint suitable for submission to Tamil Nadu government departments.

            Raw transcription: "{transcribed_text}"

            Instructions:
            1. Clean up any transcription errors or unclear phrases
            2. Structure the content into a clear subject and detailed description
            3. Use formal, respectful language appropriate for government communication
            4. Ensure the grievance is specific and actionable
            5. Add context where necessary to make the issue clear
            6. Keep the enhanced version between 100-400 words

            Format your response as:
            Subject: [Clear, concise subject line]
            Description: [Detailed, well-structured description]
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": enhancement_prompt
                    }
                ],
                max_tokens=600,
                temperature=0.5
            )
            
            enhanced_text = response.choices[0].message.content.strip()
            
            # Parse subject and description
            subject = ""
            description = ""
            
            lines = enhanced_text.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('Subject:'):
                    subject = line.replace('Subject:', '').strip()
                elif line.startswith('Description:'):
                    description = '\n'.join(lines[i:]).replace('Description:', '').strip()
                    break
            
            if not description:
                description = enhanced_text
                subject = "Audio-recorded grievance"
            
            logger.info(f"Text enhancement completed successfully")
            
            return {
                "success": True,
                "enhanced_text": description,
                "subject": subject,
                "enhancement_method": "OpenAI GPT-4"
            }
            
        except Exception as e:
            logger.error(f"Text enhancement failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "enhanced_text": transcribed_text
            }
    
    def process_image_file(self, file_content: bytes, filename: str) -> Dict:
        """
        Process uploaded image file and generate grievance text
        
        Args:
            file_content: Binary file content
            filename: Original filename
            
        Returns:
            Dict containing processing results
        """
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
                
                # Resize if too large (max 2048x2048 for API efficiency)
                max_size = 2048
                if image.width > max_size or image.height > max_size:
                    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
                # Convert back to bytes
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='JPEG', quality=85)
                processed_image_data = img_byte_arr.getvalue()
                
                # Analyze the image
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
            logger.error(f"Image processing failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def process_audio_file(self, file_content: bytes, filename: str) -> Dict:
        """
        Process uploaded audio file and transcribe to text
        
        Args:
            file_content: Binary file content
            filename: Original filename
            
        Returns:
            Dict containing processing results
        """
        try:
            # Validate file type
            allowed_extensions = ['.mp3', '.wav', '.m4a', '.ogg', '.flac']
            file_ext = os.path.splitext(filename)[1].lower()
            
            if file_ext not in allowed_extensions:
                return {
                    "success": False,
                    "error": f"Unsupported audio format: {file_ext}. Allowed: {', '.join(allowed_extensions)}"
                }
            
            # Check file size (max 25MB for Whisper API)
            max_size = 25 * 1024 * 1024  # 25MB
            if len(file_content) > max_size:
                return {
                    "success": False,
                    "error": f"Audio file too large. Maximum size is 25MB, got {len(file_content) / (1024*1024):.1f}MB"
                }
            
            # Transcribe the audio
            transcription_result = self.transcribe_audio_to_text(file_content, file_ext[1:])
            
            if transcription_result["success"]:
                # Enhance the transcribed text for grievance format
                enhancement_result = self.enhance_transcribed_text_for_grievance(
                    transcription_result["transcribed_text"]
                )
                
                # Combine results
                final_result = {
                    **transcription_result,
                    "original_filename": filename,
                    "file_size": len(file_content),
                    "enhanced_text": enhancement_result.get("enhanced_text", transcription_result["transcribed_text"]),
                    "subject": enhancement_result.get("subject", "Audio-recorded grievance"),
                    "enhancement_success": enhancement_result["success"]
                }
                
                return final_result
            else:
                return transcription_result
            
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

# Initialize AI services
ai_services = AIServices()