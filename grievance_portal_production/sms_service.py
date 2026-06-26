"""
Twilio SMS Service for Tamil Nadu Grievance Portal
Handles SMS notifications for status updates
"""

import os
from datetime import datetime
from typing import Optional, Dict, Any
import logging

try:
    from twilio.rest import Client
    from twilio.base.exceptions import TwilioException
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    Client = None
    TwilioException = Exception

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SMSService:
    """
    SMS service class using Twilio API for sending status update notifications
    """
    
    def __init__(self):
        """Initialize SMS service with Twilio credentials from environment variables"""
        self.client = None
        self.from_number = None
        self.is_configured = False
        
        if not TWILIO_AVAILABLE:
            logger.warning("Twilio package not installed. SMS notifications will be simulated.")
            return
        
        # Get Twilio credentials from environment variables
        account_sid = os.environ.get('TWILIO_ACCOUNT_SID')
        auth_token = os.environ.get('TWILIO_AUTH_TOKEN')
        self.from_number = os.environ.get('TWILIO_PHONE_NUMBER')
        
        if account_sid and auth_token and self.from_number:
            try:
                self.client = Client(account_sid, auth_token)
                self.is_configured = True
                logger.info("Twilio SMS service initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Twilio client: {str(e)}")
                self.is_configured = False
        else:
            logger.warning("Twilio credentials not found in environment variables")
            logger.warning("Required: TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER")
    
    def send_status_update_sms(self, grievance_data: Dict[str, Any], old_status: str, new_status: str) -> Dict[str, Any]:
        """
        Send SMS notification about grievance status update
        
        Args:
            grievance_data: Dictionary containing grievance information
            old_status: Previous status of the grievance
            new_status: New status of the grievance
            
        Returns:
            Dictionary with success status and message details
        """
        try:
            # Extract required information
            tracking_id = grievance_data.get('tracking_id', 'N/A')
            name = grievance_data.get('name', 'N/A')
            phone = grievance_data.get('phone', 'N/A')
            subject = grievance_data.get('petition_subject', 'N/A')
            department = grievance_data.get('department', 'N/A')
            
            # Validate phone number
            if not phone or phone == 'N/A':
                return {
                    "success": False,
                    "error": "No phone number available for SMS notification",
                    "method": "validation_error"
                }
            
            # Ensure phone number is in correct format
            formatted_phone = self._format_phone_number(phone)
            if not formatted_phone:
                return {
                    "success": False,
                    "error": f"Invalid phone number format: {phone}",
                    "method": "validation_error"
                }
            
            # Create SMS message
            sms_message = self._create_status_update_message(
                name, tracking_id, subject, old_status, new_status, department
            )
            
            # Send SMS
            if self.is_configured and self.client:
                return self._send_real_sms(formatted_phone, sms_message, tracking_id)
            else:
                return self._simulate_sms(formatted_phone, sms_message, tracking_id)
                
        except Exception as e:
            logger.error(f"Error in send_status_update_sms: {str(e)}")
            return {
                "success": False,
                "error": f"SMS service error: {str(e)}",
                "method": "exception"
            }
    
    def _format_phone_number(self, phone: str) -> Optional[str]:
        """
        Format phone number for Twilio (ensure it starts with country code)
        
        Args:
            phone: Raw phone number from user input
            
        Returns:
            Formatted phone number or None if invalid
        """
        try:
            # Remove all non-digit characters
            clean_phone = ''.join(filter(str.isdigit, phone))
            
            # Handle different Indian phone number formats
            if len(clean_phone) == 10:
                # Add India country code (+91)
                return f"+91{clean_phone}"
            elif len(clean_phone) == 12 and clean_phone.startswith('91'):
                # Already has country code
                return f"+{clean_phone}"
            elif len(clean_phone) == 13 and clean_phone.startswith('91'):
                # Has country code with extra digit
                return f"+{clean_phone[1:]}"
            else:
                logger.warning(f"Unexpected phone number format: {phone} (clean: {clean_phone})")
                # Try to use as-is if it looks like it has country code
                if len(clean_phone) >= 10:
                    return f"+{clean_phone}"
                return None
                
        except Exception as e:
            logger.error(f"Error formatting phone number {phone}: {str(e)}")
            return None
    
    def _create_status_update_message(self, name: str, tracking_id: str, subject: str, 
                                     old_status: str, new_status: str, department: str) -> str:
        """
        Create formatted SMS message for status update
        
        Args:
            name: Petitioner name
            tracking_id: Grievance tracking ID
            subject: Grievance subject
            old_status: Previous status
            new_status: Current status
            department: Department handling the grievance
            
        Returns:
            Formatted SMS message string
        """
        # Create status update message
        status_messages = {
            'pending': 'is being reviewed',
            'in_progress': 'is now under active investigation',
            'resolved': 'has been resolved',
            'rejected': 'has been reviewed and closed'
        }
        
        status_description = status_messages.get(new_status.lower(), f'status updated to {new_status}')
        
        # Keep message under 160 characters when possible for single SMS
        short_subject = subject[:50] + "..." if len(subject) > 50 else subject
        
        message = f"""TN Grievance Portal Alert

Dear {name},

Your grievance {tracking_id} {status_description}.

Subject: {short_subject}
Department: {department}

Track: portal.tn.gov.in/track

- TN Grievance Portal"""
        
        return message.strip()
    
    def _send_real_sms(self, phone: str, message: str, tracking_id: str) -> Dict[str, Any]:
        """
        Send actual SMS using Twilio API
        
        Args:
            phone: Formatted phone number
            message: SMS message content
            tracking_id: Grievance tracking ID for logging
            
        Returns:
            Result dictionary with success status and details
        """
        try:
            # Send SMS via Twilio
            twilio_message = self.client.messages.create(
                body=message,
                from_=self.from_number,
                to=phone
            )
            
            logger.info(f"SMS sent successfully for grievance {tracking_id}")
            logger.info(f"Twilio Message SID: {twilio_message.sid}")
            
            return {
                "success": True,
                "message": "SMS sent successfully via Twilio",
                "method": "twilio_api",
                "message_sid": twilio_message.sid,
                "to_number": phone,
                "from_number": self.from_number,
                "message_content": message,
                "sent_at": datetime.now().isoformat()
            }
            
        except TwilioException as te:
            logger.error(f"Twilio API error for {tracking_id}: {str(te)}")
            return {
                "success": False,
                "error": f"Twilio API error: {str(te)}",
                "method": "twilio_api_error",
                "to_number": phone,
                "message_content": message
            }
        except Exception as e:
            logger.error(f"Unexpected error sending SMS for {tracking_id}: {str(e)}")
            return {
                "success": False,
                "error": f"SMS sending failed: {str(e)}",
                "method": "unexpected_error"
            }
    
    def _simulate_sms(self, phone: str, message: str, tracking_id: str) -> Dict[str, Any]:
        """
        Simulate SMS sending when Twilio is not configured
        
        Args:
            phone: Formatted phone number
            message: SMS message content
            tracking_id: Grievance tracking ID for logging
            
        Returns:
            Result dictionary with success status and simulation details
        """
        logger.info(f"📱 SIMULATED SMS for grievance {tracking_id}:")
        logger.info(f"To: {phone}")
        logger.info(f"Message: {message}")
        logger.info("-" * 60)
        
        # Provide helpful configuration message
        config_message = "SMS simulation mode - configure Twilio credentials for real SMS delivery"
        if not TWILIO_AVAILABLE:
            config_message = "Install Twilio package: pip install twilio"
        
        return {
            "success": True,
            "message": "SMS simulated successfully (Twilio not configured)",
            "method": "simulation",
            "to_number": phone,
            "from_number": self.from_number or "Not configured",
            "message_content": message,
            "sent_at": datetime.now().isoformat(),
            "config_help": config_message,
            "required_env_vars": [
                "TWILIO_ACCOUNT_SID",
                "TWILIO_AUTH_TOKEN", 
                "TWILIO_PHONE_NUMBER"
            ]
        }
    
    def test_service(self) -> Dict[str, Any]:
        """
        Test the SMS service configuration and connectivity
        
        Returns:
            Dictionary with test results and configuration status
        """
        try:
            test_results = {
                "twilio_package_available": TWILIO_AVAILABLE,
                "client_initialized": self.client is not None,
                "credentials_configured": self.is_configured,
                "from_number": self.from_number,
                "service_status": "unknown"
            }
            
            if not TWILIO_AVAILABLE:
                test_results["service_status"] = "package_missing"
                test_results["message"] = "Twilio package not installed. Run: pip install twilio"
                return test_results
            
            if not self.is_configured:
                test_results["service_status"] = "not_configured"
                test_results["message"] = "Twilio credentials not configured in environment variables"
                test_results["required_env_vars"] = [
                    "TWILIO_ACCOUNT_SID",
                    "TWILIO_AUTH_TOKEN", 
                    "TWILIO_PHONE_NUMBER"
                ]
                return test_results
            
            # Test Twilio connection
            if self.client:
                try:
                    # Try to access account info to test credentials
                    account = self.client.api.accounts.get()
                    test_results["service_status"] = "operational"
                    test_results["message"] = "Twilio service is configured and operational"
                    test_results["account_sid"] = account.sid
                    test_results["account_status"] = account.status
                    return test_results
                except TwilioException as te:
                    test_results["service_status"] = "auth_error" 
                    test_results["message"] = f"Twilio authentication failed: {str(te)}"
                    return test_results
            
            test_results["service_status"] = "error"
            test_results["message"] = "Unknown service configuration error"
            return test_results
            
        except Exception as e:
            return {
                "service_status": "error",
                "message": f"Service test failed: {str(e)}",
                "twilio_package_available": TWILIO_AVAILABLE
            }
    
    def send_test_sms(self, test_phone: str) -> Dict[str, Any]:
        """
        Send a test SMS to verify the service is working
        
        Args:
            test_phone: Phone number to send test SMS to
            
        Returns:
            Dictionary with test SMS result
        """
        try:
            # Create test grievance data
            test_grievance = {
                'tracking_id': 'TEST-2025-SMS001',
                'name': 'Test User',
                'phone': test_phone,
                'petition_subject': 'SMS Service Test',
                'department': 'Information Technology Department'
            }
            
            # Send test status update SMS
            result = self.send_status_update_sms(
                test_grievance, 
                'pending', 
                'in_progress'
            )
            
            result["test_mode"] = True
            result["test_phone"] = test_phone
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Test SMS failed: {str(e)}",
                "test_mode": True,
                "test_phone": test_phone
            }

# Create global SMS service instance
sms_service = SMSService()

# Convenience function for easy import
def send_grievance_status_sms(grievance_data: Dict[str, Any], old_status: str, new_status: str) -> Dict[str, Any]:
    """
    Convenience function to send status update SMS
    
    Args:
        grievance_data: Complete grievance information
        old_status: Previous status
        new_status: Updated status
        
    Returns:
        SMS sending result dictionary
    """
    return sms_service.send_status_update_sms(grievance_data, old_status, new_status)