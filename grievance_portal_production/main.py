import os
from dotenv import load_dotenv
load_dotenv('backend/.env')
print(f"[DEBUG] Environment loaded - GROQ_API_KEY exists: {os.environ.get('GROQ_API_KEY') is not None}")
print(f"[DEBUG] GROQ_API_KEY length: {len(os.environ.get('GROQ_API_KEY', ''))}")
print(f"[DEBUG] TWILIO_ACCOUNT_SID: {os.environ.get('TWILIO_ACCOUNT_SID', 'NOT_FOUND')}")
print(f"[DEBUG] TWILIO_AUTH_TOKEN length: {len(os.environ.get('TWILIO_AUTH_TOKEN', ''))}")
print(f"[DEBUG] TWILIO_PHONE_NUMBER: {os.environ.get('TWILIO_PHONE_NUMBER', 'NOT_FOUND')}")

from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import mysql.connector
import bcrypt
import os
from datetime import datetime, timedelta
import requests
import difflib
from pymongo import MongoClient
from pydantic import BaseModel
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from rag_classifier import RAGEnhancedClassifier

# Import SMS service for Twilio notifications
try:
    from sms_service import send_grievance_status_sms
    SMS_SERVICE_AVAILABLE = True
    print("[DEBUG] SMS service imported successfully")
except ImportError as e:
    SMS_SERVICE_AVAILABLE = False
    print(f"[DEBUG] SMS service not available: {e}")

# Try to import AI services (both paid and free alternatives)
ai_services = None
free_ai_services = None

try:
    from ai_services import ai_services
    print("[DEBUG] OpenAI services loaded successfully")
except Exception as e:
    print(f"[DEBUG] OpenAI services not available: {e}")
    ai_services = None

try:
    from ai_services_free import free_ai_services
    print("[DEBUG] Free AI services loaded successfully")
except Exception as e:
    print(f"[DEBUG] Free AI services not available: {e}")
    free_ai_services = None

# Determine which AI service to use
def get_ai_service():
    """Get available AI service (prefer OpenAI if available, fallback to free)"""
    openai_key = os.environ.get("OPENAI_API_KEY")
    if ai_services and openai_key and openai_key != "your_openai_api_key_here":
        return ai_services, "OpenAI (Paid)"
    elif free_ai_services:
        return free_ai_services, "Free AI Services"
    else:
        return None, "No AI Services Available"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Mount static files for frontend
app.mount("/frontend", StaticFiles(directory="frontend", html=True), name="frontend")

# Mount uploads directory for serving uploaded files
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Initialize RAG classifier
rag_classifier = RAGEnhancedClassifier()

# --------------------------- DB Connection ----------------------------

def connect_to_db():
    mongo_uri = os.environ.get("MONGODB_URI")
    if not mongo_uri:
        raise RuntimeError("MONGODB_URI environment variable not set.")
    client = MongoClient(mongo_uri)
    return client.petition_db

# --------------------------- Department Mapping ----------------------------

department_tables = {
    "Adi Dravidar and Tribal Welfare Department": "petitions_adi_dravidar_tribal_welfare",
    "Agriculture and Farmers welfares Department": "petitions_agriculture_farmers_welfare",
    "Animal Husbandry and Dairying and Fisheries and Fishermen Welfare Department": "petitions_animal_husbandry_fisheries",
    "BC MBC and Minorities Welfare Department": "petitions_bc_mbc_minorities_welfare",
    "Co-operation Food and Consumer Protection Department": "petitions_cooperation_food_consumer_protection",
    "Commercial Taxes and Registration Department": "petitions_commercial_taxes_registration",
    "Energy Department": "petitions_energy",
    "Environment Climate Change and Forests Department": "petitions_environment_climate_forests",
    "Finance Department": "petitions_finance",
    "Handlooms Handicrafts Textiles and Khadi Department": "petitions_handlooms_handicrafts_textiles_khadi",
    "Health and Family Welfare Department": "petitions_health_family_welfare",
    "Higher Education Department": "petitions_higher_education",
    "Highways and Minor Ports Department": "petitions_highways_minor_ports",
    "Human Resources Management Department": "petitions_human_resources_management",
    "Home Prohibition and Excise Department": "petitions_home_prohibition_excise",
    "Housing and Urban Development Department": "petitions_housing_urban_development",
    "Industries Department": "petitions_industries",
    "Information Technology Department": "petitions_information_technology",
    "Labour Welfare and Skill Development Department": "petitions_labour_welfare_skill_development",
    "Law Department": "petitions_law",
    "Legislative Assembly Department": "petitions_legislative_assembly",
    "Micro Small and Medium Enterprises Department": "petitions_micro_small_medium_enterprises",
    "Municipal Administration and Water Supply Department": "petitions_municipal_admin_water_supply",
    "Public Elections Department": "petitions_public_elections",
    "Public Department": "petitions_public",
    "Public Works Department": "petitions_pwd",
    "Revenue and Disaster Management Department": "petitions_revenue_disaster_management",
    "Rural Development and Panchayat Raj Department": "petitions_rural_development_panchayat_raj",
    "School Education Department": "petitions_school_education",
    "Social Welfare and Women Empowerment Department": "petitions_social_welfare_women_empowerment",
    "Tamil Dev. and Information Department": "petitions_tamil_dev_information",
    "Tamil Nadu Water Supply and Drainage Board": "petitions_tn_water_supply_drainage_board",
    "Tourism Culture and Religious Endowments Department": "petitions_tourism_culture_religious_endowments",
    "Transport Department": "petitions_transport",
    "Welfare of Differently Abled Persons": "petitions_welfare_diff_abled_persons",
    "Youth Welfare and Sports Development Department": "petitions_youth_welfare_sports_development",
    "Water Resources Department": "petitions_water_resources",
    "Planning Development Department": "petitions_planning_development",
    "Special Programme Implementation": "petitions_special_programme_implementation"
}

# --------------------------- Basic Routes ----------------------------

@app.get("/")
def index():
    return {"message": "API up and running."}

# --------------------------- RAG-Enhanced Department Classification ----------------------------

def classify_with_groq_rag(petition_text):
    """Enhanced classification using RAG (Retrieval-Augmented Generation)"""
    api_key = os.environ.get("GROQ_API_KEY")
    print(f"[DEBUG] RAG Classification - API key exists: {api_key is not None}")
    if not api_key:
        print("[DEBUG] No API key found, returning Public Department")
        return "Public Department"  # fallback if no API key
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Get enhanced prompt with RAG context
    enhanced_prompt = rag_classifier.enhance_classification_prompt(petition_text)
    print(f"[DEBUG] RAG Context retrieved for: {petition_text[:50]}...")
    
    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "user", "content": enhanced_prompt}
        ],
        "temperature": 0,
        "max_tokens": 100
    }
    try:
        print(f"[DEBUG] Making RAG-enhanced request to Groq API...")
        response = requests.post(url, headers=headers, json=data, timeout=15)
        response.raise_for_status()
        result = response.json()
        department = result["choices"][0]["message"]["content"].strip()
        print(f"[DEBUG] RAG-Enhanced Groq API response: {department}")
        return department
    except Exception as e:
        print(f"[DEBUG] RAG Groq API error: {str(e)}")
        return "Public Department"  # fallback on error

# --------------------------- Department Classification ----------------------------

def classify_with_groq(petition_text):
    api_key = os.environ.get("GROQ_API_KEY")
    print(f"[DEBUG] API key exists: {api_key is not None}")
    if not api_key:
        print("[DEBUG] No API key found, returning Public Department")
        return "Public Department"  # fallback if no API key
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    prompt = (
        "You are a classifier that assigns a single, most appropriate Tamil Nadu government department to each petition, using the list below. Direct mapping from the petition text to the department's functional domain is essential; if you're unsure, choose the closest possible match. Avoid returning \"General,\" \"Unknown,\" or any response outside the specified options. To improve classification accuracy, consider the core responsibilities of each department as described in the official list of Tamil Nadu departments.\n"
        "Wikipedia\n"
        "List of Tamil Nadu Government Departments:\n"
        "Adi Dravidar and Tribal Welfare Department\n"
        "Agriculture and Farmers Welfare Department\n"
        "Animal Husbandry and Dairying and Fisheries and Fishermen Welfare Department\n"
        "BC MBC and Minorities Welfare Department\n"
        "Co-operation Food and Consumer Protection Department\n"
        "Commercial Taxes and Registration Department\n"
        "Energy Department\n"
        "Environment Climate Change and Forests Department\n"
        "Finance Department\n"
        "Handlooms Handicrafts Textiles and Khadi Department\n"
        "Health and Family Welfare Department\n"
        "Higher Education Department\n"
        "Highways and Minor Ports Department\n"
        "Human Resources Management Department\n"
        "Home Prohibition and Excise Department\n"
        "Housing and Urban Development Department\n"
        "Industries Department\n"
        "Information Technology Department\n"
        "Labour Welfare and Skill Development Department\n"
        "Law Department\n"
        "Legislative Assembly Department\n"
        "Micro Small and Medium Enterprises Department\n"
        "Municipal Administration and Water Supply Department\n"
        "Public Elections Department\n"
        "Public Department\n"
        "Public Works Department\n"
        "Revenue and Disaster Management Department\n"
        "Rural Development and Panchayat Raj Department\n"
        "School Education Department\n"
        "Social Welfare and Women Empowerment Department\n"
        "Tamil Dev. and Information Department\n"
        "Tamil Nadu Water Supply and Drainage Board\n"
        "Tourism Culture and Religious Endowments Department\n"
        "Transport Department\n"
        "Welfare of Differently Abled Persons\n"
        "Youth Welfare and Sports Development Department\n"
        "Water Resources Department\n"
        "Planning Development Department\n"
        "Special Programme Implementation\n"
        "Instructions:\n"
        "When classifying, match the petition’s theme—like “educational facilities,” “water supply issues,” “forest conservation,” “IT infrastructure,” or “tax filing problem”—with a department that holds responsibility for related services. Avoid vague categories like “Public” unless absolutely necessary.\n"
        f"Petition: '{petition_text}'\nDepartment:"
    )
    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "max_tokens": 50
    }
    try:
        print(f"[DEBUG] Making request to Groq API...")
        response = requests.post(url, headers=headers, json=data, timeout=10)
        response.raise_for_status()
        result = response.json()
        department = result["choices"][0]["message"]["content"].strip()
        print(f"[DEBUG] Groq API response: {department}")
        return department
    except Exception as e:
        print(f"[DEBUG] Groq API error: {str(e)}")
        return "Public Department"  # fallback on error

# --------------------------- Cosine Similarity Confidence ----------------------------

def compute_cosine_confidence(petition_text, classified_department):
    """
    Compute confidence score using cosine similarity between petition text
    and classified department's information from knowledge base
    """
    try:
        from rag_knowledge_base import DEPARTMENT_KNOWLEDGE_BASE
        
        # Get department information
        dept_info = DEPARTMENT_KNOWLEDGE_BASE.get(classified_department, {})
        if not dept_info:
            return 0.4  # Default low confidence for unknown departments
        
        # Combine all department text information
        dept_text_parts = []
        
        # Add description
        if 'description' in dept_info:
            dept_text_parts.append(dept_info['description'])
            
        # Add responsibilities
        if 'responsibilities' in dept_info:
            dept_text_parts.extend(dept_info['responsibilities'])
            
        # Add example grievances
        if 'example_grievances' in dept_info:
            dept_text_parts.extend(dept_info['example_grievances'])
            
        # Add keywords
        if 'keywords' in dept_info:
            dept_text_parts.extend(dept_info['keywords'])
        
        # Combine into single department text
        department_text = ' '.join(dept_text_parts).lower()
        petition_text_clean = petition_text.lower()
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams for better context
            min_df=1
        )
        
        # Fit and transform both texts
        corpus = [petition_text_clean, department_text]
        tfidf_matrix = vectorizer.fit_transform(corpus)
        
        # Calculate cosine similarity
        similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # Normalize and adjust the score
        # Apply sigmoid-like transformation for better score distribution
        import math
        confidence = 1 / (1 + math.exp(-10 * (similarity_score - 0.1)))  # Sigmoid centered at 0.1
        
        # Ensure confidence is in valid range
        confidence = max(0.05, min(0.95, confidence))  # Clamp between 5% and 95%
        
        return confidence
        
    except Exception as e:
        print(f"Cosine confidence calculation error: {e}")
        return 0.5  # Default medium confidence on error

# --------------------------- Enhanced RAG + Groq Classification ----------------------------

def classify_with_rag_groq(petition_text):
    """
    PRIMARY CLASSIFICATION METHOD: RAG + Groq LLM Combined
    1. RAG provides contextual knowledge from department database
    2. Groq LLM makes intelligent classification using enhanced context
    3. Always attempts to provide meaningful classification
    """
    try:
        # STEP 1: Get relevant departmental context from RAG
        try:
            relevant_context = rag_classifier.get_relevant_context(petition_text, top_k=5)  # Increased context
            print(f"[DEBUG] RAG context retrieved: {len(relevant_context) if relevant_context else 0} matches")
        except Exception as rag_error:
            print(f"[DEBUG] RAG context error: {rag_error}")
            relevant_context = []
        
        if not relevant_context:
            print("Warning: No RAG context found, using basic classification")
            
        # STEP 2: Enhance prompt with RAG context for Groq
        try:
            enhanced_prompt = rag_classifier.enhance_classification_prompt(petition_text)
            print(f"[DEBUG] Enhanced prompt created successfully")
        except Exception as prompt_error:
            print(f"[DEBUG] Prompt enhancement error: {prompt_error}")
            # Fallback to basic prompt
            enhanced_prompt = f"Classify this petition text to appropriate Tamil Nadu department: {petition_text}"
        
        # STEP 3: Call Groq API with RAG-enhanced prompt
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("Warning: No Groq API key, falling back to RAG-only")
            if relevant_context and len(relevant_context) > 0:
                top_dept = relevant_context[0].get("department", "Public Department")
                top_score = relevant_context[0].get("similarity_score", 0.5)
                return top_dept, top_score, "RAG-only classification (no API key)"
            return "Public Department", 0.0, "No API key and no RAG context"
        
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "llama-3.1-8b-instant",  # Reliable working model
            "messages": [
                {
                    "role": "system", 
                    "content": "You are an expert classifier for Tamil Nadu government departments. Use the provided context to make accurate classifications. Return ONLY the department name, no explanations."
                },
                {
                    "role": "user", 
                    "content": enhanced_prompt
                }
            ],
            "temperature": 0.1,  # Low temperature for consistent results
            "max_tokens": 50,    # Limit to department name only
            "top_p": 0.9
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=20)
            response.raise_for_status()
            result = response.json()
            
            department = result["choices"][0]["message"]["content"].strip()
            print(f"[DEBUG] Groq API successful: {department}")
            
            # STEP 4: Calculate confidence score using cosine similarity
            confidence = compute_cosine_confidence(petition_text, department)
            
            # Additional confidence adjustments based on RAG context
            if relevant_context and len(relevant_context) > 0:
                # Find if the classified department matches top RAG suggestions
                rag_dept_names = [ctx.get("department", "") for ctx in relevant_context]
                if department in rag_dept_names:
                    # Boost confidence if RAG also suggests this department
                    rag_index = rag_dept_names.index(department)
                    rag_boost = 0.1 * (1 - rag_index * 0.02)  # Higher boost for top matches
                    confidence = min(confidence + rag_boost, 0.95)
                
            explanation = {
                "method": "RAG + Groq combined with cosine similarity",
                "rag_matches": len(relevant_context) if relevant_context else 0,
                "top_rag_dept": relevant_context[0].get("department", "None") if relevant_context else "None",
                "top_rag_score": relevant_context[0].get("similarity_score", 0.0) if relevant_context else 0.0,
                "llm_classification": department,
                "cosine_confidence": confidence,
                "confidence_method": "TF-IDF cosine similarity with department knowledge base"
            }
            
            return department, confidence, explanation
            
        except requests.exceptions.RequestException as e:
            print(f"Groq API error: {e}")
            # FALLBACK 1: Use RAG-only classification with cosine confidence
            if relevant_context and len(relevant_context) > 0:
                rag_department = relevant_context[0].get("department", "Public Department")
                cosine_confidence = compute_cosine_confidence(petition_text, rag_department)
                explanation = {
                    "method": "RAG-only fallback with cosine confidence (API error)",
                    "rag_matches": len(relevant_context),
                    "top_rag_dept": rag_department,
                    "top_rag_score": relevant_context[0].get("similarity_score", 0.0),
                    "cosine_confidence": cosine_confidence
                }
                return rag_department, cosine_confidence, explanation
            
    except Exception as e:
        print(f"RAG-Groq classification error: {e}")
        # FALLBACK 2: Basic RAG classification with cosine confidence
        try:
            relevant_context = rag_classifier.get_relevant_context(petition_text, top_k=1)
            if relevant_context and len(relevant_context) > 0:
                rag_department = relevant_context[0].get("department", "Public Department")
                cosine_confidence = compute_cosine_confidence(petition_text, rag_department) * 0.8  # Reduce for fallback
                explanation = {
                    "method": "RAG-only fallback with cosine confidence (system error)",
                    "rag_matches": len(relevant_context),
                    "top_rag_dept": rag_department,
                    "cosine_confidence": cosine_confidence
                }
                return rag_department, cosine_confidence, explanation
        except Exception as rag_error:
            print(f"Final RAG fallback error: {rag_error}")
    
    # LAST RESORT: Default classification
    return "Public Department", 0.1, "Default fallback classification"

# --------------------------- Auth: Register ----------------------------

@app.post("/register")
def register_account(
    full_name: str = Form(...),
    new_user: str = Form(...),
    new_pass: str = Form(...)
):
    db = connect_to_db()
    users_collection = db["users"]  # Access the 'users' collection

    # Check if the username already exists
    if users_collection.find_one({"username": new_user}):
        return {"error": "Username is already taken"}

    # Hash the password and insert the new user
    hashed_password = bcrypt.hashpw(new_pass.encode(), bcrypt.gensalt()).decode()
    users_collection.insert_one({
        "name": full_name,
        "username": new_user,
        "password": hashed_password
    })

    return {"message": "User registered successfully"}

# --------------------------- Auth: Login ----------------------------

@app.post("/login")
def login_user(user_id: str = Form(...), passcode: str = Form(...)):
    predefined_admins = {
        "pwd": {"password": "123", "dashboard": "officer_dashboard.html", "department": "Public Works Department"},
        "fin": {"password": "123", "dashboard": "officer_dashboard.html", "department": "Finance Department"},
        "edu": {"password": "123", "dashboard": "officer_dashboard.html", "department": "Education Department"},
        "adi": {"password": "123", "dashboard": "officer_dashboard.html", "department": "Adi Dravidar and Tribal Welfare Department"},
        "agr": {"password": "123", "dashboard": "officer_dashboard.html", "department": "Agriculture and Farmers welfares Department"},
        "ani": {"password": "123", "dashboard": "officer_dashboard.html", "department": "Animal Husbandry and Dairying and Fisheries and Fishermen Welfare Department"},
        "bcm": {"password": "123", "dashboard": "officer_dashboard.html", "department": "BC MBC and Minorities Welfare Department"},
        "cof": {"password": "123", "dashboard": "officer_dashboard.html", "department": "Co-operation Food and Consumer Protection Department"},
        "ctr": {"password": "123", "dashboard": "officer_dashboard.html", "department": "Commercial Taxes and Registration Department"},
        "ene": {"password": "123", "dashboard": "officer_dashboard.html", "department": "Energy Department"},
        "env": {"password": "123", "dashboard": "officer_dashboard.html", "department": "Environment Climate Change and Forests Department"},
        "han": {"password": "123", "dashboard": "officer_dashboard.html", "department": "Handlooms Handicrafts Textiles and Khadi Department"},
        "hea": {"password": "123", "dashboard": "officer_dashboard.html", "department": "Health and Family Welfare Department"},
        "hed": {"password": "123", "dashboard": "officer_dashboard.html", "department": "Higher Education Department"},
        "hig": {"password": "123", "dashboard": "officer_dashboard.html", "department": "Highways and Minor Ports Department"},
        "hrm": {"password": "123", "dashboard": "officer_dashboard.html", "department": "Human Resources Management Department"},
        "hom": {"password": "123", "dashboard": "officer_dashboard.html", "department": "Home Prohibition and Excise Department"},
        "hou": {"password": "123", "dashboard": "officer_dashboard.html", "department": "Housing and Urban Development Department"},
        "ind": {"password": "123", "dashboard": "officer_dashboard.html", "department": "Industries Department"},
        "itd": {"password": "123", "dashboard": "officer_dashboard.html", "department": "Information Technology Department"},
        "lab": {"password": "123", "dashboard": "officer_dashboard.html", "department": "Labour Welfare and Skill Development Department"},
        "law": {"password": "123", "dashboard": "officer_dashboard.html", "department": "Law Department"},
        "leg": {"password": "123", "dashboard": "officer_dashboard.html", "department": "Legislative Assembly Department"},
        "mic": {"password": "123", "dashboard": "officer_dashboard.html", "department": "Micro Small and Medium Enterprises Department"},
        "mun": {"password": "123", "dashboard": "officer_dashboard.html", "department": "Municipal Administration and Water Supply Department"},
        "pel": {"password": "123", "dashboard": "officer_dashboard.html", "department": "Public Elections Department"},
        "pub": {"password": "123", "dashboard": "officer_dashboard.html", "department": "Public Department"},
        "rev": {"password": "123", "dashboard": "officer_dashboard.html", "department": "Revenue and Disaster Management Department"},
        "rur": {"password": "123", "dashboard": "officer_dashboard.html", "department": "Rural Development and Panchayat Raj Department"},
        "sch": {"password": "123", "dashboard": "officer_dashboard.html", "department": "School Education Department"},
        "soc": {"password": "123", "dashboard": "officer_dashboard.html", "department": "Social Welfare and Women Empowerment Department"},
        "tam": {"password": "123", "dashboard": "officer_dashboard.html", "department": "Tamil Dev. and Information Department"},
        "tou": {"password": "123", "dashboard": "officer_dashboard.html", "department": "Tourism Culture and Religious Endowments Department"},
        "tra": {"password": "123", "dashboard": "officer_dashboard.html", "department": "Transport Department"},
        "wel": {"password": "123", "dashboard": "officer_dashboard.html", "department": "Welfare of Differently Abled Persons"},
        "you": {"password": "123", "dashboard": "officer_dashboard.html", "department": "Youth Welfare and Sports Development Department"},
        "wat": {"password": "123", "dashboard": "officer_dashboard.html", "department": "Water Resources Department"},
        "pla": {"password": "123", "dashboard": "officer_dashboard.html", "department": "Planning Development Department"},
        "spe": {"password": "123", "dashboard": "officer_dashboard.html", "department": "Special Programme Implementation"},
        "tws": {"password": "123", "dashboard": "officer_dashboard.html", "department": "Tamil Nadu Water Supply and Drainage Board"},
        "master": {"password": "123", "dashboard": "masterpage.html", "department": "Master Administrator"}
    }

    # Admin login
    if user_id in predefined_admins and predefined_admins[user_id]["password"] == passcode:
        return {
            "message": "Admin login successful",
            "role": "admin",
            "dashboard": predefined_admins[user_id]["dashboard"],
            "department": predefined_admins[user_id]["department"]
        }

    # User login
    db = connect_to_db()
    users_collection = db["users"]  # Access the 'users' collection
    user = users_collection.find_one({"username": user_id})  # Query the user by username

    if user and bcrypt.checkpw(passcode.encode(), user["password"].encode()):
        return {
            "message": "User login successful",
            "role": "user",
            "dashboard": "dashboardaph.html"
        }

    return {"error": "Invalid login credentials"}

# --------------------------- Classify Petition ----------------------------

@app.post("/classify")
def predict_category(petition_text: str = Form(...)):
    print(f"[DEBUG] Input petition text: {petition_text}")
    
    # Use enhanced RAG + Groq classification with cosine similarity
    department_raw, confidence, explanation = classify_with_rag_groq(petition_text)
    print(f"[DEBUG] RAG + Groq raw output: {department_raw}")
    print(f"[DEBUG] RAG + Groq confidence: {confidence:.3f}")

    if not department_raw:
        print("[DEBUG] No department from RAG + Groq, using Public Department fallback")
        return {"category": "Public Department", "confidence": 0.0, "method": "fallback"}

    department_clean = department_raw.strip().lower()
    print(f"[DEBUG] Cleaned department: {department_clean}")

    # Try exact match
    for dept in department_tables:
        if dept.lower() == department_clean:
            print(f"[DEBUG] Exact match found: {dept}")
            return {
                "category": dept, 
                "confidence": confidence, 
                "method": "rag_groq_exact_match",
                "explanation": explanation
            }

    # Try partial match (e.g., if LLM returns 'School Education Department' but you store 'Education Department')
    for dept in department_tables:
        if dept.lower() in department_clean or department_clean in dept.lower():
            print(f"[DEBUG] Partial match found: {dept}")
            return {
                "category": dept, 
                "confidence": confidence, 
                "method": "rag_groq_partial_match",
                "explanation": explanation
            }

    # Fuzzy match
    import difflib
    match = difflib.get_close_matches(department_clean, [d.lower() for d in department_tables.keys()], n=1, cutoff=0.6)
    if match:
        for dept in department_tables:
            if dept.lower() == match[0]:
                print(f"[DEBUG] Fuzzy match found: {dept}")
                return {
                    "category": dept, 
                    "confidence": confidence * 0.8,  # Reduce confidence for fuzzy match
                    "method": "rag_groq_fuzzy_match",
                    "explanation": explanation
                }

    print(f"[DEBUG] No match found for '{department_raw}', using Public Department fallback")
    return {
        "category": "Public Department", 
        "confidence": 0.1, 
        "method": "fallback",
        "explanation": explanation
    }

# --------------------------- Compare Classification Methods ----------------------------

@app.post("/classify_compare")
def compare_classification_methods(petition_text: str = Form(...)):
    """Compare RAG-enhanced vs standard classification"""
    print(f"[DEBUG] Comparing classification methods for: {petition_text}")
    
    # Standard classification (old method)
    old_result = classify_with_groq(petition_text)
    
    # Enhanced RAG + Groq classification (new method)
    new_dept, confidence, explanation = classify_with_rag_groq(petition_text)
    
    return {
        "input": petition_text,
        "old_classification": old_result,
        "new_classification": new_dept,
        "confidence": confidence,
        "detailed_explanation": explanation,
        "comparison": {
            "same_result": old_result.strip() == new_dept.strip(),
            "new_advantage": confidence > 0.6,
            "method_used": explanation.get("method", "unknown")
        }
    }

# --------------------------- Submit Petition ----------------------------

@app.post("/submit_to_department")
async def save_petition(
    name: str = Form(...),
    phone: str = Form(...),
    address: str = Form(...),
    petition_type: str = Form(...),
    petition_subject: str = Form(...),
    petition_description: str = Form(...),
    category: str = Form(...),
    attachment: UploadFile = File(None),
    latitude: str = Form(None),
    longitude: str = Form(None),
    location_accuracy: str = Form(None),
    location_timestamp: str = Form(None),
    location_capture_method: str = Form(None)
):
    """
    Submit a petition to the appropriate department
    
    This endpoint:
    1. Saves the grievance details to the appropriate department collection
    2. Automatically assigns a priority level based on text content
    3. Checks for similar grievances using TF-IDF cosine similarity
    4. Initializes timeline tracking
    5. Returns the assigned department, priority level, and similarity results
    
    Priority is determined by scanning for keywords like:
    "urgent", "danger", "accident", "emergency", "fire", "violence", 
    "death", "injury", "threat", "hospital"
    
    Args:
        name: Petitioner's name
        phone: Contact phone number
        address: Petitioner's address
        petition_type: Type of grievance
        petition_subject: Brief subject line
        petition_description: Detailed description of grievance
        category: Department category
        
    Returns:
        Status message, assigned department, priority level, and similarity results
    """
    try:
        # Normalize category for lookup
        category_clean = category.strip()
        table = department_tables.get(category_clean)
        if not table:
            # Try case-insensitive match
            for key in department_tables:
                if key.lower() == category_clean.lower():
                    table = department_tables[key]
                    category_clean = key
                    break
        if not table:
            return {"error": f"Invalid or undefined category: {category}"}
        
        db = connect_to_db()
        petitions_collection = db[table]
        
        # Detect priority level based on subject and description
        combined_text = f"{petition_subject} {petition_description}"
        priority_level = detect_priority(combined_text)
        
        # Check for similar grievances
        similar_grievances = find_similar_grievances(combined_text, category_clean)
        
        # Generate a unique tracking ID
        tracking_id = generate_tracking_id()
        
        # Ensure the tracking ID is unique across all departments
        is_unique = False
        attempts = 0
        while not is_unique and attempts < 10:
            # Check if this tracking ID already exists in any department
            exists = False
            for dept_table in department_tables.values():
                dept_collection = db[dept_table]
                if dept_collection.find_one({"tracking_id": tracking_id}):
                    exists = True
                    break
            
            if not exists:
                is_unique = True
            else:
                tracking_id = generate_tracking_id()
                attempts += 1
        
        # Initialize timeline with submission entry
        initial_timeline = [{
            'timestamp': datetime.now(),
            'date': datetime.now().strftime("%d-%b-%Y"),
            'time': datetime.now().strftime("%H:%M:%S"),
            'status': 'pending',
            'comment': 'Grievance submitted successfully',
            'update_type': 'submission'
        }]
        
        # Handle file attachment if present
        attachment_info = None
        if attachment and attachment.filename:
            try:
                # Validate file type
                allowed_types = ['image/jpeg', 'image/png', 'image/gif', 'image/bmp', 'application/pdf']
                if attachment.content_type not in allowed_types:
                    return {"error": f"Unsupported file type. Allowed: JPEG, PNG, GIF, BMP, PDF"}
                
                # Validate file size (max 2MB)
                file_content = await attachment.read()
                if len(file_content) > 2 * 1024 * 1024:  # 2MB limit
                    return {"error": "File size too large. Maximum allowed size is 2MB"}
                
                # Generate unique filename
                import uuid
                file_extension = attachment.filename.split('.')[-1].lower()
                unique_filename = f"{tracking_id}_{uuid.uuid4().hex}.{file_extension}"
                file_path = os.path.join("uploads", unique_filename)
                
                # Save file to uploads directory
                with open(file_path, "wb") as buffer:
                    buffer.write(file_content)
                
                attachment_info = {
                    "original_filename": attachment.filename,
                    "stored_filename": unique_filename,
                    "file_path": file_path,
                    "file_size": len(file_content),
                    "content_type": attachment.content_type,
                    "upload_date": datetime.now().strftime("%d-%b-%Y %H:%M:%S")
                }
                
            except Exception as file_error:
                print(f"File handling error: {file_error}")
                return {"error": f"Failed to process attachment: {str(file_error)}"}
        
        # Prepare location data if provided
        location_data = None
        if latitude and longitude:
            try:
                # Determine capture method
                capture_method = location_capture_method if location_capture_method else "unknown"
                
                location_data = {
                    "latitude": float(latitude),
                    "longitude": float(longitude),
                    "accuracy": float(location_accuracy) if location_accuracy else None,
                    "timestamp": location_timestamp if location_timestamp else datetime.now().isoformat(),
                    "coordinates_string": f"{latitude}, {longitude}",
                    "capture_method": capture_method
                }
                print(f"Location data prepared: {capture_method} - {latitude}, {longitude}")
            except ValueError:
                print(f"Invalid location data: lat={latitude}, lng={longitude}")
                location_data = None
        
        # Prepare petition data
        petition_data = {
            "tracking_id": tracking_id,
            "name": name,
            "phone": phone,
            "address": address,
            "petition_type": petition_type,
            "petition_subject": petition_subject,
            "petition_description": petition_description,
            "status": "pending",
            "priority": priority_level,
            "created_at": datetime.now().strftime("%d-%b-%Y"),
            "department": category_clean,
            "timeline": initial_timeline,
            "last_updated": datetime.now(),
            "attachment": attachment_info,  # Add attachment info to petition data
            "location": location_data  # Add location data
        }
        
        # Add related_to field if similar grievances found
        if similar_grievances:
            petition_data["related_to"] = [g['grievance_id'] for g in similar_grievances]
            petition_data["similarity_detected"] = True
        else:
            petition_data["similarity_detected"] = False
        
        # Insert the petition
        petitions_collection.insert_one(petition_data)
        
        # Prepare response
        response_data = {
            "message": "Petition recorded successfully", 
            "department": category_clean,
            "priority": priority_level,
            "tracking_id": tracking_id,
            "similarity_detected": len(similar_grievances) > 0,
            "similar_grievances_count": len(similar_grievances)
        }
        
        # Include similarity details if found
        if similar_grievances:
            response_data["similar_grievances"] = similar_grievances
            response_data["similarity_message"] = f"Found {len(similar_grievances)} similar grievance(s). Your issue may be related to existing cases."
        
        return response_data
        
    except OSError as os_err:
        return {"error": f"File system error: {os_err}"}
    except Exception as ex:
        return {"error": f"Unexpected error: {ex}"}

# --------------------------- Admin View Petitions ----------------------------

@app.get("/admin/petitions")
def list_petitions(department: str):
    table = department_tables.get(department)
    if not table:
        return {"error": "Invalid department requested"}

    db = connect_to_db()
    petitions_collection = db[table]  # Access the collection for the department
    result = list(petitions_collection.find())  # Retrieve all documents from the collection

    # Convert MongoDB documents to JSON-serializable format and add tracking IDs where missing
    for petition in result:
        petition["_id"] = str(petition["_id"])
        
        # If petition doesn't have a tracking_id (for old records), generate one
        if not petition.get("tracking_id"):
            tracking_id = generate_tracking_id()
            # Update the record in the database
            petitions_collection.update_one(
                {"_id": petition["_id"]}, 
                {"$set": {"tracking_id": tracking_id}}
            )
            petition["tracking_id"] = tracking_id

    return result

@app.get("/admin/petitions/by_priority")
def list_petitions_by_priority(department: str, priority: str = None):
    """
    List petitions for a department, optionally filtered by priority level
    
    Args:
        department: Department name
        priority: Optional priority filter ("High", "Medium", "Low")
        
    Returns:
        List of petitions matching the criteria
    """
    table = department_tables.get(department)
    if not table:
        return {"error": "Invalid department requested"}

    db = connect_to_db()
    petitions_collection = db[table]
    
    # Build query filter
    query = {}
    if priority:
        query["priority"] = priority
    
    # Execute the query
    result = list(petitions_collection.find(query))
    
    # Convert MongoDB documents to JSON-serializable format and add tracking IDs where missing
    for petition in result:
        petition["_id"] = str(petition["_id"])
        
        # If petition doesn't have a tracking_id (for old records), generate one
        if not petition.get("tracking_id"):
            tracking_id = generate_tracking_id()
            # Update the record in the database
            petitions_collection.update_one(
                {"_id": petition["_id"]}, 
                {"$set": {"tracking_id": tracking_id}}
            )
            petition["tracking_id"] = tracking_id

    return result

# --------------------------- Track Grievance ----------------------------

@app.post("/track_grievance")
def track_grievance(grievance_id: str = Form(...), phone: str = Form(...)):
    """
    Track a grievance using the new tracking system
    
    This function searches for grievances using the stored tracking_id field
    Phone number is used for additional verification
    """
    try:
        db = connect_to_db()
        
        # Validate inputs
        if not grievance_id or grievance_id.strip() == "":
            return {"found": False, "message": "Please provide your grievance tracking ID to track your grievance."}
        
        if not phone or phone.strip() == "":
            return {"found": False, "message": "Please provide your phone number for verification."}
        
        # Clean the grievance ID
        grievance_id = grievance_id.strip().upper()
        phone = phone.strip()
        
        print(f"[DEBUG] Tracking grievance with ID: {grievance_id} and phone: {phone}")
        
        # Search across all department tables for the tracking ID
        found_petition = None
        found_department = None
        
        for department, table_name in department_tables.items():
            collection = db[table_name]
            
            # Search by tracking_id
            petition = collection.find_one({"tracking_id": grievance_id})
            
            if petition:
                # Verify phone number matches for security
                if petition.get("phone") == phone:
                    found_petition = petition
                    found_department = department
                    print(f"[DEBUG] Found grievance {grievance_id} in {department}")
                    break
                else:
                    print(f"[DEBUG] Found grievance {grievance_id} but phone number doesn't match")
                    return {
                        "found": False, 
                        "message": "The phone number doesn't match the one used to file this grievance. Please check your phone number."
                    }
        
        if not found_petition:
            print(f"[DEBUG] No grievance found with tracking ID: {grievance_id}")
            # Check if there are any grievances for this phone number
            user_grievances = []
            for department, table_name in department_tables.items():
                collection = db[table_name]
                petitions = list(collection.find({"phone": phone}))
                for pet in petitions:
                    if pet.get("tracking_id"):
                        user_grievances.append(f"{pet['tracking_id']} ({department})")
            
            if user_grievances:
                return {
                    "found": False,
                    "message": f"No grievance found with ID '{grievance_id}'. Your registered grievances are: {', '.join(user_grievances)}"
                }
            else:
                return {
                    "found": False,
                    "message": "No grievance found with the provided tracking ID and phone number."
                }
        
        # Convert ObjectId to string for JSON serialization
        found_petition["_id"] = str(found_petition["_id"])
        
        # Add department information
        found_petition["department"] = found_department
        
        # Use the stored tracking_id as the display ID
        found_petition["id"] = found_petition.get("tracking_id", grievance_id)
        
        # Generate timeline updates
        created_date = found_petition.get("created_at", datetime.now().strftime("%d-%b-%Y"))
        
        updates = [
            {
                "date": created_date,
                "title": "Grievance Received",
                "description": "Your grievance has been received and registered in the system."
            },
            {
                "date": created_date,
                "title": "Assigned to Department",
                "description": f"Your grievance has been assigned to {found_department}."
            }
        ]
        
        # Add status-specific updates
        status = found_petition.get("status", "pending").lower()
        if status == "in_progress":
            updates.append({
                "date": datetime.now().strftime("%d-%b-%Y"),
                "title": "Under Review",
                "description": "Your grievance is currently being reviewed by the department."
            })
        elif status == "resolved":
            updates.append({
                "date": datetime.now().strftime("%d-%b-%Y"),
                "title": "Resolved",
                "description": "Your grievance has been resolved."
            })
        elif status == "rejected":
            updates.append({
                "date": datetime.now().strftime("%d-%b-%Y"),
                "title": "Rejected",
                "description": "Your grievance has been reviewed and rejected."
            })
        
        found_petition["updates"] = updates
        
        print(f"[DEBUG] Successfully retrieved grievance {grievance_id}")
        return {"found": True, "grievance": found_petition}
        
    except Exception as ex:
        print(f"[ERROR] Exception in track_grievance: {str(ex)}")
        return {"error": f"An error occurred while tracking your grievance: {str(ex)}"}

@app.post("/update_grievance_status")
def update_grievance_status(
    grievance_id: str = Form(...), 
    status: str = Form(...), 
    department: str = Form(...),
    comment: str = Form(None)
):
    """
    Update the status of a grievance with timeline tracking and notifications
    """
    try:
        db = connect_to_db()
        
        # Validate status
        valid_statuses = ["pending", "resolved", "rejected", "in_progress"]
        if status.lower() not in valid_statuses:
            return {"success": False, "message": "Invalid status value"}
            
        # Get the department table name
        table_name = department_tables.get(department)
        if not table_name:
            return {"success": False, "message": "Invalid department"}
            
        collection = db[table_name]
        
        # Find the petition using the tracking_id
        petition = collection.find_one({"tracking_id": grievance_id})
        
        if not petition:
            return {"success": False, "message": "Grievance not found with the provided tracking ID"}
        
        # Get the old status for notifications
        old_status = petition.get('status', 'unknown')
        new_status = status.lower()
        
        # Create timeline entry
        timeline_entry = {
            'timestamp': datetime.now(),
            'date': datetime.now().strftime("%d-%b-%Y"),
            'time': datetime.now().strftime("%H:%M:%S"),
            'status': new_status,
            'comment': comment if comment else f"Status updated to {new_status}",
            'update_type': 'status_update'
        }
        
        # Update the status and add timeline entry
        update_result = collection.update_one(
            {"tracking_id": grievance_id},
            {
                "$set": {
                    "status": new_status,
                    "last_updated": datetime.now()
                },
                "$push": {"timeline": timeline_entry}
            }
        )
        
        if update_result.modified_count == 0:
            return {"success": False, "message": "Failed to update status"}
        
        # Send notification to petitioner if status changed
        if old_status != new_status:
            try:
                send_notification_to_petitioner(petition, old_status, new_status)
            except Exception as e:
                print(f"Failed to send notification for {grievance_id}: {str(e)}")
                # Don't fail the status update if notification fails
        
        current_date = datetime.now().strftime("%d-%b-%Y")
        status_title = status.capitalize()
        status_description = comment if comment else f"Status updated to {status}"
        
        return {
            "success": True,
            "message": "Status updated successfully",
            "timeline_update": {
                "date": current_date,
                "title": status_title,
                "description": status_description
            },
            "notification_sent": True
        }
        
    except Exception as ex:
        return {"success": False, "message": f"An error occurred: {str(ex)}"}

# --------------------------- Utility Functions ----------------------------

import random
import string

def generate_tracking_id():
    """
    Generate a unique tracking ID in the format GR-YYYY-XXXXXX
    Where YYYY is the current year and XXXXXX is a 6-digit alphanumeric code
    """
    year = datetime.now().year
    # Generate a 6-character alphanumeric code (uppercase letters and numbers)
    code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    return f"GR-{year}-{code}"

def detect_priority(text: str) -> str:
    """
    Detect priority level based on text content
    Returns: "High", "Medium", or "Low"
    """
    text_lower = text.lower()
    
    # High priority keywords
    high_priority_keywords = [
        'urgent', 'emergency', 'critical', 'immediate', 'life threatening',
        'danger', 'death', 'accident', 'fire', 'flood', 'earthquake',
        'medical emergency', 'hospital', 'ambulance', 'police',
        'violence', 'harassment', 'threat', 'safety'
    ]
    
    # Medium priority keywords
    medium_priority_keywords = [
        'important', 'asap', 'soon', 'quick', 'fast', 'priority',
        'water shortage', 'power outage', 'road damage', 'broken',
        'not working', 'complaint', 'problem', 'issue'
    ]
    
    # Check for high priority
    for keyword in high_priority_keywords:
        if keyword in text_lower:
            return "High"
    
    # Check for medium priority
    for keyword in medium_priority_keywords:
        if keyword in text_lower:
            return "Medium"
    
    # Default to Medium for general grievances
    return "Medium"

# --------------------------- Similarity Detection ---------------------------

def find_similar_grievances(petition_text, department, similarity_threshold=0.8):
    """
    Find similar grievances using TF-IDF and cosine similarity
    
    Args:
        petition_text: The text to compare (subject + description)
        department: Department to search within
        similarity_threshold: Minimum similarity score (default 0.8 for 80%)
        
    Returns:
        List of similar grievances with their similarity scores
    """
    try:
        db = connect_to_db()
        table_name = department_tables.get(department)
        if not table_name:
            return []
            
        collection = db[table_name]
        
        # Get all existing grievances in this department
        existing_grievances = list(collection.find({}, {
            'petition_subject': 1, 
            'petition_description': 1, 
            'tracking_id': 1,
            '_id': 1
        }))
        
        if len(existing_grievances) < 1:
            return []
        
        # Prepare texts for comparison
        texts = []
        grievance_refs = []
        
        # Add the new petition text
        combined_new_text = f"{petition_text}".lower().strip()
        texts.append(combined_new_text)
        grievance_refs.append(None)  # Placeholder for new petition
        
        # Add existing grievances
        for grievance in existing_grievances:
            subject = grievance.get('petition_subject', '')
            description = grievance.get('petition_description', '')
            combined_text = f"{subject} {description}".lower().strip()
            
            if combined_text:  # Only add non-empty texts
                texts.append(combined_text)
                grievance_refs.append(grievance)
        
        if len(texts) < 2:  # Need at least 2 texts for comparison
            return []
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2),  # Include unigrams and bigrams
            min_df=1,
            max_df=0.95
        )
        
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Calculate cosine similarity between new petition and all existing ones
        similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        
        # Find similar grievances above threshold
        similar_grievances = []
        for i, score in enumerate(similarity_scores):
            if score >= similarity_threshold:
                grievance = grievance_refs[i + 1]  # +1 because we skip the new petition
                if grievance:
                    similar_grievances.append({
                        'grievance_id': grievance.get('tracking_id', str(grievance['_id'])),
                        'similarity_score': float(score),
                        'subject': grievance.get('petition_subject', 'N/A'),
                        'description': grievance.get('petition_description', 'N/A')[:100] + '...'
                    })
        
        # Sort by similarity score (highest first)
        similar_grievances.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return similar_grievances
        
    except Exception as e:
        print(f"Error in similarity detection: {str(e)}")
        return []

# --------------------------- Timeline Management ---------------------------

def add_timeline_entry(grievance_id, department, status, comment="", update_type="status_update"):
    """
    Add a timeline entry to a grievance
    
    Args:
        grievance_id: Tracking ID of the grievance
        department: Department name
        status: New status
        comment: Optional comment
        update_type: Type of update (status_update, comment, reminder, etc.)
    """
    try:
        db = connect_to_db()
        table_name = department_tables.get(department)
        if not table_name:
            return False
            
        collection = db[table_name]
        
        # Create timeline entry
        timeline_entry = {
            'timestamp': datetime.now(),
            'date': datetime.now().strftime("%d-%b-%Y"),
            'time': datetime.now().strftime("%H:%M:%S"),
            'status': status,
            'comment': comment,
            'update_type': update_type
        }
        
        # Add timeline entry to the grievance
        result = collection.update_one(
            {'tracking_id': grievance_id},
            {
                '$push': {'timeline': timeline_entry},
                '$set': {'last_updated': datetime.now()}
            }
        )
        
        return result.modified_count > 0
        
    except Exception as e:
        print(f"Error adding timeline entry: {str(e)}")
        return False

def get_grievance_timeline(grievance_id, department):
    """
    Get the complete timeline for a grievance
    """
    try:
        db = connect_to_db()
        table_name = department_tables.get(department)
        if not table_name:
            return []
            
        collection = db[table_name]
        grievance = collection.find_one({'tracking_id': grievance_id}, {'timeline': 1})
        
        if grievance and 'timeline' in grievance:
            timeline = grievance['timeline']
            # Convert datetime objects to strings for JSON serialization
            for entry in timeline:
                if isinstance(entry.get('timestamp'), datetime):
                    entry['timestamp'] = entry['timestamp'].isoformat()
            return timeline
        
        return []
        
    except Exception as e:
        print(f"Error getting timeline: {str(e)}")
        return []

# --------------------------- Notification System ---------------------------

def send_notification_to_petitioner(grievance_data, old_status, new_status):
    """
    Send notifications to petitioners via SMS (Twilio) and Email simulation
    
    Args:
        grievance_data: Complete grievance information
        old_status: Previous status
        new_status: Updated status
    """
    try:
        tracking_id = grievance_data.get('tracking_id', 'N/A')
        name = grievance_data.get('name', 'N/A')
        phone = grievance_data.get('phone', 'N/A')
        subject = grievance_data.get('petition_subject', 'N/A')
        
        # Result tracking for different notification methods
        notification_results = {
            'sms_result': None,
            'email_result': None,
            'overall_success': False
        }
        
        # Send SMS notification via Twilio
        if SMS_SERVICE_AVAILABLE:
            try:
                sms_result = send_grievance_status_sms(grievance_data, old_status, new_status)
                notification_results['sms_result'] = sms_result
                
                if sms_result.get('success', False):
                    print(f"✅ SMS notification sent successfully for {tracking_id}")
                    print(f"📱 SMS Details: {sms_result.get('message', 'Sent via Twilio')}")
                    if sms_result.get('method') == 'simulation':
                        print(f"📋 SMS Content: {sms_result.get('message_content', '')}")
                else:
                    print(f"❌ SMS notification failed for {tracking_id}: {sms_result.get('error', 'Unknown error')}")
                    
            except Exception as sms_error:
                print(f"❌ SMS service error for {tracking_id}: {str(sms_error)}")
                notification_results['sms_result'] = {
                    'success': False,
                    'error': f"SMS service exception: {str(sms_error)}",
                    'method': 'exception'
                }
        else:
            print(f"⚠️ SMS service not available for {tracking_id} - using fallback notification")
            notification_results['sms_result'] = {
                'success': False,
                'error': 'SMS service not configured',
                'method': 'service_unavailable'
            }
        
        # Simulate Email notification (existing functionality)
        try:
            email_message = f"""
Subject: Grievance Status Update - {tracking_id}

Dear {name},

This is to inform you that your grievance has been updated:

Tracking ID: {tracking_id}
Subject: {subject}
Previous Status: {old_status.upper()}
Current Status: {new_status.upper()}
Updated On: {datetime.now().strftime("%d-%b-%Y %H:%M:%S")}

You can track your grievance status at: portal.tn.gov.in/track

Best regards,
Tamil Nadu Grievance Portal Team
            """.strip()
            
            # Log the email notification (simulating actual sending)
            print(f"� EMAIL NOTIFICATION SIMULATED for {tracking_id}:")
            print(email_message)
            print("-" * 60)
            
            notification_results['email_result'] = {
                'success': True,
                'method': 'simulation',
                'content': email_message
            }
            
        except Exception as email_error:
            print(f"❌ Email notification error for {tracking_id}: {str(email_error)}")
            notification_results['email_result'] = {
                'success': False,
                'error': str(email_error),
                'method': 'exception'
            }
        
        # Store comprehensive notification log in database
        try:
            db = connect_to_db()
            notification_log = {
                'grievance_id': tracking_id,
                'recipient_name': name,
                'recipient_phone': phone,
                'notification_type': 'status_update',
                'old_status': old_status,
                'new_status': new_status,
                'sent_at': datetime.now(),
                'sms_details': notification_results['sms_result'],
                'email_details': notification_results['email_result'],
                'sms_success': notification_results['sms_result'].get('success', False) if notification_results['sms_result'] else False,
                'email_success': notification_results['email_result'].get('success', False) if notification_results['email_result'] else False
            }
            
            # Add SMS content to log if available
            if notification_results['sms_result'] and notification_results['sms_result'].get('message_content'):
                notification_log['sms_content'] = notification_results['sms_result']['message_content']
            
            # Add email content to log if available
            if notification_results['email_result'] and notification_results['email_result'].get('content'):
                notification_log['email_content'] = notification_results['email_result']['content']
            
            db.notification_logs.insert_one(notification_log)
            print(f"📝 Notification log saved for {tracking_id}")
            
        except Exception as log_error:
            print(f"⚠️ Failed to save notification log for {tracking_id}: {str(log_error)}")
        
        # Determine overall success
        sms_success = notification_results['sms_result'].get('success', False) if notification_results['sms_result'] else False
        email_success = notification_results['email_result'].get('success', False) if notification_results['email_result'] else False
        
        notification_results['overall_success'] = sms_success or email_success
        
        # Print summary
        if notification_results['overall_success']:
            methods = []
            if sms_success:
                methods.append("SMS")
            if email_success:
                methods.append("Email")
            print(f"✅ Notification sent successfully for {tracking_id} via: {', '.join(methods)}")
        else:
            print(f"❌ All notification methods failed for {tracking_id}")
        
        return notification_results['overall_success']
        
    except Exception as e:
        print(f"💥 Critical error in notification system for {tracking_id}: {str(e)}")
        return False

# --------------------------- FastAPI Startup/Shutdown Events ---------------------------

@app.on_event("startup")
async def startup_event():
    """Initialize the Grievance Portal API"""
    print("Grievance Portal API started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup when the app shuts down"""
    print("Grievance Portal API stopped")

# --------------------------- Timeline and Similarity Endpoints ---------------------------

@app.get("/grievance/timeline")
def get_timeline(tracking_id: str, department: str):
    """
    Get the complete timeline for a specific grievance
    """
    try:
        timeline = get_grievance_timeline(tracking_id, department)
        return {
            "success": True,
            "timeline": timeline
        }
    except Exception as e:
        return {"success": False, "message": f"Error retrieving timeline: {str(e)}"}

@app.get("/grievance/similar")
def check_similar_grievances(department: str, text: str, threshold: float = 0.8):
    """
    Check for similar grievances (useful for testing or manual checks)
    """
    try:
        similar = find_similar_grievances(text, department, threshold)
        return {
            "success": True,
            "similar_grievances": similar,
            "count": len(similar)
        }
    except Exception as e:
        return {"success": False, "message": f"Error checking similarity: {str(e)}"}

@app.get("/admin/notifications")
def get_notification_logs(limit: int = 50):
    """
    Get notification logs for administrative purposes
    """
    try:
        db = connect_to_db()
        notifications = list(db.notification_logs.find().sort("sent_at", -1).limit(limit))
        
        # Convert ObjectId and datetime to strings for JSON serialization
        for notification in notifications:
            notification["_id"] = str(notification["_id"])
            if isinstance(notification.get("sent_at"), datetime):
                notification["sent_at"] = notification["sent_at"].isoformat()
        
        return {
            "success": True,
            "notifications": notifications
        }
    except Exception as e:
        return {"success": False, "message": f"Error retrieving notifications: {str(e)}"}

@app.post("/admin/test_similarity")
def test_similarity_detection():
    """
    Test endpoint to verify similarity detection is working
    """
    try:
        # Test with sample data
        test_text = "Water supply problem in my area. No water for 3 days."
        department = "Tamil Nadu Water Supply and Drainage Board"
        
        similar = find_similar_grievances(test_text, department, 0.5)  # Lower threshold for testing
        
        return {
            "success": True,
            "test_text": test_text,
            "department": department,
            "similar_found": len(similar),
            "similar_grievances": similar,
            "message": "Similarity detection is working properly"
        }
    except Exception as e:
        return {"success": False, "message": f"Similarity test failed: {str(e)}"}

@app.post("/admin/test_notifications")
def test_notification_system():
    """
    Test endpoint to verify notification system is working
    """
    try:
        # Create test grievance data
        test_grievance = {
            'tracking_id': 'TEST-2024-ABC123',
            'name': 'Test User',
            'phone': '+91 9876543210',
            'petition_subject': 'Test Notification System'
        }
        
        # Send test notification
        success = send_notification_to_petitioner(test_grievance, 'pending', 'in_progress')
        
        return {
            "success": success,
            "message": "Test notification sent successfully" if success else "Test notification failed"
        }
    except Exception as e:
        return {"success": False, "message": f"Notification test failed: {str(e)}"}

# --------------------------- Master Statistics Endpoint ---------------------------

@app.get("/admin/statistics")
def get_master_statistics():
    """
    Get comprehensive statistics across all departments for the master dashboard
    """
    try:
        db = connect_to_db()
        
        stats = {
            "total_grievances": 0,
            "by_department": {},
            "by_status": {"pending": 0, "resolved": 0, "rejected": 0, "in_progress": 0},
            "by_priority": {"High": 0, "Medium": 0, "Low": 0},
            "recent_activity": {"last_7_days": 0, "last_30_days": 0},
            "departments_count": len(department_tables),
            "generated_at": datetime.now().isoformat()
        }
        
        # Calculate date thresholds
        now = datetime.now()
        week_ago = now - timedelta(days=7)
        month_ago = now - timedelta(days=30)
        
        # Aggregate data from all departments
        for department, table_name in department_tables.items():
            try:
                collection = db[table_name]
                
                # Get all grievances for this department
                grievances = list(collection.find({}))
                
                dept_stats = {
                    "total": len(grievances),
                    "pending": 0,
                    "resolved": 0,
                    "rejected": 0,
                    "in_progress": 0,
                    "high_priority": 0,
                    "medium_priority": 0,
                    "low_priority": 0,
                    "recent_7_days": 0,
                    "recent_30_days": 0
                }
                
                for grievance in grievances:
                    # Count by status
                    status = grievance.get("status", "pending").lower()
                    if status in dept_stats:
                        dept_stats[status] += 1
                        stats["by_status"][status] += 1
                    
                    # Count by priority
                    priority = grievance.get("priority", "Medium")
                    if priority in ["High", "Medium", "Low"]:
                        dept_stats[f"{priority.lower()}_priority"] += 1
                        stats["by_priority"][priority] += 1
                    
                    # Count recent activity
                    created_at = grievance.get("created_at")
                    if created_at:
                        try:
                            if isinstance(created_at, str):
                                # Parse date string
                                created_date = datetime.strptime(created_at, "%d-%b-%Y")
                            elif isinstance(created_at, datetime):
                                created_date = created_at
                            else:
                                continue
                            
                            if created_date >= week_ago:
                                dept_stats["recent_7_days"] += 1
                                stats["recent_activity"]["last_7_days"] += 1
                            if created_date >= month_ago:
                                dept_stats["recent_30_days"] += 1
                                stats["recent_activity"]["last_30_days"] += 1
                        except:
                            pass
                
                stats["by_department"][department] = dept_stats
                stats["total_grievances"] += dept_stats["total"]
                
            except Exception as e:
                print(f"Error processing statistics for {department}: {str(e)}")
                continue
        
        return {
            "success": True,
            "statistics": stats
        }
        
    except Exception as e:
        print(f"Error generating master statistics: {str(e)}")
        return {"success": False, "message": f"Error retrieving statistics: {str(e)}"}

# --------------------------- AI-Powered Image and Audio Processing Endpoints ---------------------------

@app.post("/analyze_image")
async def analyze_image_for_grievance(file: UploadFile = File(...)):
    """
    Analyze uploaded image using AI to generate grievance text
    
    This endpoint:
    1. Validates the uploaded image file
    2. Uses OpenAI Vision API to analyze the image
    3. Generates appropriate grievance text based on visual content
    4. Returns structured data with subject, description, and confidence
    
    Args:
        file: Uploaded image file (JPEG, PNG, GIF, BMP)
        
    Returns:
        JSON response with generated grievance text and analysis details
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read file content
        file_content = await file.read()
        
        # Check file size (max 10MB for images)
        max_size = 10 * 1024 * 1024  # 10MB
        if len(file_content) > max_size:
            raise HTTPException(
                status_code=400, 
                detail=f"Image file too large. Maximum size is 10MB, got {len(file_content) / (1024*1024):.1f}MB"
            )
        
        # Get available AI service
        service, service_name = get_ai_service()
        if not service:
            raise HTTPException(status_code=503, detail="AI services not available. Please configure OpenAI API key or install free AI dependencies.")
        
        # Process the image
        result = service.process_image_file(file_content, file.filename)
        
        if result["success"]:
            return {
                "success": True,
                "message": f"Image analyzed successfully using {service_name}",
                "data": {
                    "generated_subject": result.get("subject", ""),
                    "generated_description": result.get("generated_text", ""),
                    "confidence": result.get("confidence", 0.0),
                    "word_count": result.get("word_count", 0),
                    "analysis_method": result.get("analysis_method", service_name),
                    "service_used": service_name,
                    "file_info": {
                        "filename": result.get("original_filename", file.filename),
                        "size": result.get("file_size", len(file_content)),
                        "dimensions": result.get("image_dimensions", "unknown")
                    }
                }
            }
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Image analysis failed"))
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Image analysis endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/analyze_photo")
async def analyze_photo_for_petition(image: UploadFile = File(...)):
    """
    Analyze uploaded photo and generate petition suggestions
    
    This endpoint specifically handles photo analysis for the petition form,
    returning suggested subject and description for grievances.
    """
    try:
        # Validate file type
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read file content
        file_content = await image.read()
        
        # Check file size (max 5MB for photos)
        max_size = 5 * 1024 * 1024  # 5MB
        if len(file_content) > max_size:
            raise HTTPException(
                status_code=400, 
                detail=f"Photo file too large. Maximum size is 5MB"
            )
        
        # Get available AI service
        service, service_name = get_ai_service()
        if not service:
            return {
                "suggested_subject": "Unable to analyze photo - AI services not available",
                "suggested_description": "Please manually describe the issue shown in the photo. Include details about what you see, where it occurred, and how it affects you or your community."
            }
        
        # Process the image for grievance generation
        result = service.process_image_file(file_content, image.filename)
        
        if result["success"]:
            subject = result.get("subject", "Grievance regarding issue in uploaded photo")
            description = result.get("generated_text", "Please describe the issue shown in the photo.")
            
            # Ensure we have meaningful content
            if not subject or len(subject.strip()) < 10:
                subject = "Grievance regarding issue shown in photo"
            
            if not description or len(description.strip()) < 20:
                description = "Based on the uploaded photo, please provide additional details about the issue, including when it occurred, how it affects you, and what resolution you seek."
            
            return {
                "suggested_subject": subject,
                "suggested_description": description,
                "confidence": result.get("confidence", 0.5),
                "service_used": service_name
            }
        else:
            return {
                "suggested_subject": "Issue requiring attention (please edit this subject)",
                "suggested_description": f"Based on the uploaded photo, there appears to be an issue that requires attention. Please provide specific details about: 1) What exactly is the problem shown in the image, 2) When and where this issue occurred, 3) How it affects you or your community, 4) What resolution or action you are seeking. Error analyzing photo: {result.get('error', 'Unknown error')}"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Photo analysis error: {str(e)}")
        return {
            "suggested_subject": "Grievance requiring attention (please edit this subject)",
            "suggested_description": f"Unable to analyze the uploaded photo due to technical issues. Please manually describe the issue shown in the image, including specific details about what the problem is, when and where it occurred, and what resolution you seek."
        }

@app.post("/transcribe_audio")
async def transcribe_audio_for_grievance(file: UploadFile = File(...)):
    """
    Transcribe uploaded audio to text and enhance for grievance submission
    
    This endpoint:
    1. Validates the uploaded audio file
    2. Uses OpenAI Whisper API to transcribe speech to text
    3. Enhances the raw transcription for formal grievance format
    4. Returns structured grievance text with subject and description
    
    Args:
        file: Uploaded audio file (MP3, WAV, M4A, OGG, FLAC)
        
    Returns:
        JSON response with transcribed and enhanced grievance text
    """
    try:
        print(f"[DEBUG] Received audio file: {file.filename}, Content-Type: {file.content_type}")
        
        # Validate file type
        allowed_audio_types = [
            'audio/mpeg', 'audio/mp3', 'audio/wav', 'audio/wave', 
            'audio/x-wav', 'audio/m4a', 'audio/ogg', 'audio/flac'
        ]
        
        if file.content_type not in allowed_audio_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported audio format: {file.content_type}. Supported: MP3, WAV, M4A, OGG, FLAC"
            )
        
        # Read file content
        file_content = await file.read()
        print(f"[DEBUG] File size: {len(file_content)} bytes")
        
        # Get available AI service
        service, service_name = get_ai_service()
        print(f"[DEBUG] Using AI service: {service_name}")
        
        if not service:
            raise HTTPException(status_code=503, detail="AI services not available. Please configure OpenAI API key or install free AI dependencies.")
        
        # Process the audio
        print(f"[DEBUG] Processing audio with {service_name}...")
        result = service.process_audio_file(file_content, file.filename)
        print(f"[DEBUG] Audio processing result: {result.get('success', False)}")
        
        if result["success"]:
            return {
                "success": True,
                "message": f"Audio transcribed successfully using {service_name}",
                "data": {
                    "transcribed_text": result.get("transcribed_text", ""),
                    "enhanced_subject": result.get("subject", ""),
                    "enhanced_description": result.get("enhanced_text", ""),
                    "confidence": result.get("confidence", 0.0),
                    "word_count": result.get("word_count", 0),
                    "transcription_method": result.get("transcription_method", service_name),
                    "service_used": service_name,
                    "enhancement_success": result.get("enhancement_success", False),
                    "file_info": {
                        "filename": result.get("original_filename", file.filename),
                        "size": result.get("file_size", len(file_content))
                    }
                }
            }
        else:
            print(f"[DEBUG] Audio processing failed: {result.get('error', 'Unknown error')}")
            raise HTTPException(status_code=500, detail=result.get("error", "Audio transcription failed"))
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"[DEBUG] Audio transcription endpoint error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/ai_generate_grievance")
async def ai_generate_grievance_text(
    text_input: str = Form(None),
    file: UploadFile = File(None)
):
    """
    Combined endpoint for AI-powered grievance generation from text, image, or audio
    
    This endpoint can handle:
    1. Text enhancement for better grievance format
    2. Image analysis to generate grievance text
    3. Audio transcription and enhancement
    
    Args:
        text_input: Optional text to enhance
        file: Optional image or audio file to process
        
    Returns:
        JSON response with generated/enhanced grievance content
    """
    try:
        if not text_input and not file:
            raise HTTPException(status_code=400, detail="Either text_input or file must be provided")
        
        result_data = {}
        
        # Handle file upload (image or audio)
        if file:
            file_content = await file.read()
            
            if file.content_type.startswith('image/'):
                # Process image
                result = ai_services.process_image_file(file_content, file.filename)
                if result["success"]:
                    result_data = {
                        "type": "image_analysis",
                        "subject": result.get("subject", ""),
                        "description": result.get("generated_text", ""),
                        "confidence": result.get("confidence", 0.0),
                        "source": "Image Analysis"
                    }
                else:
                    raise HTTPException(status_code=500, detail=result.get("error", "Image processing failed"))
                    
            elif file.content_type.startswith('audio/'):
                # Process audio
                result = ai_services.process_audio_file(file_content, file.filename)
                if result["success"]:
                    result_data = {
                        "type": "audio_transcription",
                        "subject": result.get("subject", ""),
                        "description": result.get("enhanced_text", ""),
                        "confidence": result.get("confidence", 0.0),
                        "transcribed_text": result.get("transcribed_text", ""),
                        "source": "Audio Transcription"
                    }
                else:
                    raise HTTPException(status_code=500, detail=result.get("error", "Audio processing failed"))
            else:
                raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # Handle text input enhancement
        elif text_input:
            enhancement_result = ai_services.enhance_transcribed_text_for_grievance(text_input)
            if enhancement_result["success"]:
                result_data = {
                    "type": "text_enhancement",
                    "subject": enhancement_result.get("subject", ""),
                    "description": enhancement_result.get("enhanced_text", ""),
                    "confidence": 0.8,  # Default confidence for text enhancement
                    "original_text": text_input,
                    "source": "Text Enhancement"
                }
            else:
                result_data = {
                    "type": "text_passthrough",
                    "subject": "User Input",
                    "description": text_input,
                    "confidence": 0.5,
                    "source": "Direct Input"
                }
        
        return {
            "success": True,
            "message": "Grievance content generated successfully",
            "data": result_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"AI grievance generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/ai_status")
def get_ai_services_status():
    """
    Check the status of AI services (both OpenAI and free alternatives)
    Enhanced for admin dashboard monitoring
    """
    try:
        # Check available services
        service, service_name = get_ai_service()
        
        # Check OpenAI specifically
        openai_available = False
        openai_key_exists = os.environ.get("OPENAI_API_KEY") is not None
        openai_key_valid = openai_key_exists and os.environ.get("OPENAI_API_KEY") != "your_openai_api_key_here"
        
        if ai_services and openai_key_valid:
            try:
                openai_available = ai_services.client is not None
            except:
                openai_available = False
        
        # Check free services in detail
        free_services_available = free_ai_services is not None
        
        # Check individual free service components
        image_analysis_details = {}
        audio_transcription_details = {}
        text_enhancement_details = {}
        dependencies_details = {}
        
        if free_services_available:
            try:
                # Check image analysis components
                image_analysis_details = {
                    "available": hasattr(free_ai_services, 'ocr_reader') and free_ai_services.ocr_reader is not None,
                    "ocr_engine": "EasyOCR",
                    "captioning_model": "BLIP (Salesforce/blip-image-captioning-base)",
                    "supported_formats": ["JPEG", "PNG", "GIF", "BMP"]
                }
                
                # Check audio transcription components
                audio_transcription_details = {
                    "available": hasattr(free_ai_services, 'speech_recognizer') and free_ai_services.speech_recognizer is not None,
                    "engine": "Google Speech Recognition (Free tier)",
                    "audio_processing": "pydub + ffmpeg",
                    "supported_formats": ["WAV", "MP3", "M4A", "OGG", "FLAC"]
                }
                
                # Check text enhancement
                text_enhancement_details = {
                    "available": True,
                    "method": "Rule-based formatting and enhancement",
                    "classification": "Keyword-based issue detection",
                    "features": ["Auto-capitalization", "Punctuation", "Formal language"]
                }
                
                # Check system dependencies
                import sys
                import subprocess
                
                try:
                    import torch
                    gpu_available = torch.cuda.is_available()
                except:
                    gpu_available = False
                
                try:
                    # Check ffmpeg availability
                    result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=5)
                    ffmpeg_available = result.returncode == 0
                except:
                    ffmpeg_available = False
                
                dependencies_details = {
                    "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                    "ffmpeg": ffmpeg_available,
                    "gpu_support": gpu_available,
                    "all_available": ffmpeg_available  # Main dependency for audio processing
                }
                
            except Exception as e:
                print(f"Error checking free service details: {e}")
        
        # Determine overall status
        if openai_available:
            status = "operational"
            primary_service = "OpenAI (Paid)"
        elif free_services_available:
            status = "operational"
            primary_service = "Free AI Services"
        else:
            status = "error"
            primary_service = "None"
        
        return {
            "success": True,
            "ai_services": {
                "primary_service": primary_service,
                "status": status,
                "openai_available": openai_available,
                "openai_key_configured": openai_key_exists,
                "openai_key_valid": openai_key_valid,
                "free_services_available": free_services_available,
                "image_analysis": image_analysis_details,
                "audio_transcription": audio_transcription_details,
                "text_enhancement": text_enhancement_details,
                "dependencies": dependencies_details
            },
            "service_details": {
                "openai_features": {
                    "image_analysis": "GPT-4 Vision API",
                    "audio_transcription": "Whisper API",
                    "cost": "Paid per usage"
                },
                "free_features": {
                    "image_analysis": "EasyOCR + BLIP + Rule-based",
                    "audio_transcription": "Google Speech Recognition (Free tier)",
                    "cost": "Completely free"
                }
            },
            "supported_features": {
                "image_formats": ["JPEG", "PNG", "GIF", "BMP"],
                "audio_formats": ["MP3", "WAV", "M4A", "OGG", "FLAC"],
                "max_image_size": "10MB",
                "max_audio_size": "25MB" if openai_available else "10MB"
            },
            "recommendations": {
                "for_best_quality": "Configure OpenAI API key for highest accuracy",
                "for_free_usage": "Free services provide good quality at no cost",
                "setup_help": "See AI_FEATURES_DOCUMENTATION.md for setup instructions"
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "ai_services": {
                "status": "error"
            }
        }

@app.get("/ai_logs")
def get_ai_service_logs():
    """
    Get AI service usage logs and statistics for admin dashboard
    """
    try:
        # In a real implementation, you'd fetch this from a logging database
        # For now, we'll return mock data that represents typical usage
        
        # Simulate recent activity (in production, fetch from actual logs)
        recent_logs = [
            {
                "timestamp": "2025-01-21 10:30:15",
                "service": "Image Analysis",
                "message": "Successfully analyzed grievance image with OCR",
                "status": "success"
            },
            {
                "timestamp": "2025-01-21 10:25:42",
                "service": "Audio Transcription",
                "message": "Transcribed 2.3MB audio file in 3.2 seconds",
                "status": "success"
            },
            {
                "timestamp": "2025-01-21 10:20:11",
                "service": "Text Enhancement",
                "message": "Enhanced grievance description with formal language",
                "status": "success"
            },
            {
                "timestamp": "2025-01-21 10:15:33",
                "service": "Audio Transcription",
                "message": "Audio conversion failed - ffmpeg not available",
                "status": "warning"
            },
            {
                "timestamp": "2025-01-21 10:10:22",
                "service": "Image Analysis",
                "message": "OCR detected infrastructure issue keywords",
                "status": "success"
            }
        ]
        
        # Simulate usage statistics
        stats = {
            "image_requests": 45,
            "audio_requests": 23,
            "text_enhancements": 67,
            "error_count": 3,
            "success_rate": 95.2
        }
        
        return {
            "success": True,
            "stats": stats,
            "recent_logs": recent_logs
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "stats": {},
            "recent_logs": []
        }

@app.post("/test_image_service")
def test_image_analysis_service():
    """
    Test the image analysis service functionality
    """
    try:
        service, service_name = get_ai_service()
        
        if not service:
            return {
                "success": False,
                "error": "No AI service available"
            }
        
        # Test with a simple check of the service initialization
        if hasattr(service, 'ocr_reader') and service.ocr_reader is not None:
            return {
                "success": True,
                "message": f"Image analysis service ({service_name}) is operational",
                "service": service_name,
                "features": ["OCR text extraction", "Image captioning", "Issue detection"]
            }
        else:
            return {
                "success": False,
                "error": "Image analysis components not properly initialized"
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/test_audio_service")
def test_audio_transcription_service():
    """
    Test the audio transcription service functionality
    """
    try:
        service, service_name = get_ai_service()
        
        if not service:
            return {
                "success": False,
                "error": "No AI service available"
            }
        
        # Test speech recognition initialization
        if hasattr(service, 'speech_recognizer') and service.speech_recognizer is not None:
            return {
                "success": True,
                "message": f"Audio transcription service ({service_name}) is operational",
                "service": service_name,
                "features": ["Speech-to-text", "Multiple audio formats", "Text enhancement"]
            }
        else:
            return {
                "success": False,
                "error": "Audio transcription components not properly initialized"
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/test_text_service")
def test_text_enhancement_service():
    """
    Test the text enhancement service functionality
    """
    try:
        from pydantic import BaseModel
        
        class TextRequest(BaseModel):
            text: str = "This is a test grievance"
        
        service, service_name = get_ai_service()
        
        if not service:
            return {
                "success": False,
                "error": "No AI service available"
            }
        
        # Test with default text
        test_text = "This is a test grievance"
        
        # Test text enhancement
        if hasattr(service, 'enhance_transcribed_text_for_grievance'):
            result = service.enhance_transcribed_text_for_grievance(test_text)
            
            if result.get("success", True):
                return {
                    "success": True,
                    "message": f"Text enhancement service ({service_name}) is operational",
                    "service": service_name,
                    "test_input": test_text,
                    "enhanced_output": result.get("enhanced_text", test_text),
                    "features": ["Formal language", "Proper punctuation", "Subject generation"]
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "Text enhancement failed")
                }
        else:
            return {
                "success": False,
                "error": "Text enhancement not available"
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/check_dependencies")
def check_system_dependencies():
    """
    Check system dependencies for AI services
    """
    try:
        import sys
        import subprocess
        
        dependencies = {}
        
        # Check Python version
        dependencies["python_version"] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
        # Check ffmpeg
        try:
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=5)
            dependencies["ffmpeg"] = result.returncode == 0
            if dependencies["ffmpeg"]:
                # Extract version info
                lines = result.stdout.split('\n')
                if lines:
                    version_line = lines[0]
                    dependencies["ffmpeg_version"] = version_line
            else:
                dependencies["ffmpeg_version"] = "Not installed"
        except Exception as e:
            dependencies["ffmpeg"] = False
            dependencies["ffmpeg_version"] = f"Error: {str(e)}"
        
        # Check GPU support
        try:
            import torch
            dependencies["gpu_support"] = torch.cuda.is_available()
            if dependencies["gpu_support"]:
                dependencies["gpu_count"] = torch.cuda.device_count()
                dependencies["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown"
            else:
                dependencies["gpu_count"] = 0
                dependencies["gpu_name"] = "CPU only"
        except ImportError:
            dependencies["gpu_support"] = False
            dependencies["gpu_count"] = 0
            dependencies["gpu_name"] = "PyTorch not available"
        except Exception as e:
            dependencies["gpu_support"] = False
            dependencies["gpu_count"] = 0
            dependencies["gpu_name"] = f"Error: {str(e)}"
        
        # Check critical Python packages
        try:
            import easyocr
            dependencies["easyocr"] = True
            dependencies["easyocr_version"] = getattr(easyocr, '__version__', 'Unknown')
        except ImportError:
            dependencies["easyocr"] = False
            dependencies["easyocr_version"] = "Not installed"
        
        try:
            import speech_recognition
            dependencies["speech_recognition"] = True
            dependencies["speech_recognition_version"] = getattr(speech_recognition, '__version__', 'Unknown')
        except ImportError:
            dependencies["speech_recognition"] = False
            dependencies["speech_recognition_version"] = "Not installed"
        
        try:
            import pydub
            dependencies["pydub"] = True
            dependencies["pydub_version"] = getattr(pydub, '__version__', 'Unknown')
        except ImportError:
            dependencies["pydub"] = False
            dependencies["pydub_version"] = "Not installed"
        
        # Determine overall health
        critical_deps = dependencies["easyocr"] and dependencies["speech_recognition"]
        dependencies["all_available"] = critical_deps
        dependencies["health_score"] = sum([
            dependencies.get("ffmpeg", False),
            dependencies.get("gpu_support", False),
            dependencies.get("easyocr", False),
            dependencies.get("speech_recognition", False),
            dependencies.get("pydub", False)
        ]) / 5 * 100
        
        return {
            "success": True,
            "dependencies": dependencies
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "dependencies": {}
        }

# --------------------------- SMS Service Management Endpoints ---------------------------

@app.get("/admin/sms_service_status")
def get_sms_service_status():
    """
    Get the current status of the SMS service for admin monitoring
    """
    try:
        if SMS_SERVICE_AVAILABLE:
            from sms_service import sms_service
            test_result = sms_service.test_service()
            
            return {
                "success": True,
                "sms_service": {
                    "available": True,
                    "configured": test_result.get("credentials_configured", False),
                    "status": test_result.get("service_status", "unknown"),
                    "message": test_result.get("message", ""),
                    "twilio_package": test_result.get("twilio_package_available", False),
                    "from_number": test_result.get("from_number", "Not configured"),
                    "account_status": test_result.get("account_status", "Unknown")
                },
                "configuration": {
                    "required_env_vars": [
                        "TWILIO_ACCOUNT_SID",
                        "TWILIO_AUTH_TOKEN", 
                        "TWILIO_PHONE_NUMBER"
                    ],
                    "setup_instructions": "Add Twilio credentials to your environment variables or .env file"
                }
            }
        else:
            return {
                "success": False,
                "sms_service": {
                    "available": False,
                    "configured": False,
                    "status": "service_unavailable",
                    "message": "SMS service module not available"
                },
                "configuration": {
                    "installation_required": "pip install twilio",
                    "setup_instructions": "Install Twilio package and configure credentials"
                }
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "sms_service": {
                "available": False,
                "status": "error"
            }
        }

@app.post("/admin/test_sms")
def test_sms_service(test_phone: str = Form(...)):
    """
    Send a test SMS to verify the service is working
    
    Args:
        test_phone: Phone number to send test SMS to
    """
    try:
        if not SMS_SERVICE_AVAILABLE:
            return {
                "success": False,
                "error": "SMS service not available. Install Twilio: pip install twilio"
            }
        
        from sms_service import sms_service
        
        # Validate phone number format
        if not test_phone or len(test_phone.strip()) < 10:
            return {
                "success": False,
                "error": "Please provide a valid phone number (at least 10 digits)"
            }
        
        # Send test SMS
        result = sms_service.send_test_sms(test_phone.strip())
        
        return {
            "success": result.get("success", False),
            "message": result.get("message", "Test SMS completed"),
            "details": result,
            "test_phone": test_phone
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Test SMS failed: {str(e)}",
            "test_phone": test_phone
        }

@app.get("/admin/sms_logs")
def get_sms_notification_logs(limit: int = 50):
    """
    Get SMS notification logs for administrative purposes
    """
    try:
        db = connect_to_db()
        
        # Query notifications with SMS details
        notifications = list(
            db.notification_logs.find(
                {"sms_details": {"$exists": True}},
                {
                    "grievance_id": 1,
                    "recipient_name": 1,
                    "recipient_phone": 1,
                    "old_status": 1,
                    "new_status": 1,
                    "sent_at": 1,
                    "sms_details": 1,
                    "sms_success": 1
                }
            ).sort("sent_at", -1).limit(limit)
        )
        
        # Convert ObjectId and datetime to strings for JSON serialization
        for notification in notifications:
            notification["_id"] = str(notification["_id"])
            if isinstance(notification.get("sent_at"), datetime):
                notification["sent_at"] = notification["sent_at"].isoformat()
        
        # Calculate statistics
        total_sms = len(notifications)
        successful_sms = sum(1 for n in notifications if n.get("sms_success", False))
        success_rate = (successful_sms / total_sms * 100) if total_sms > 0 else 0
        
        return {
            "success": True,
            "sms_logs": notifications,
            "statistics": {
                "total_sms_attempts": total_sms,
                "successful_sms": successful_sms,
                "failed_sms": total_sms - successful_sms,
                "success_rate": round(success_rate, 2)
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Error retrieving SMS logs: {str(e)}",
            "sms_logs": [],
            "statistics": {}
        }

@app.post("/admin/sms_configuration_help")
def get_sms_configuration_help():
    """
    Provide detailed SMS service configuration instructions
    """
    return {
        "success": True,
        "configuration_guide": {
            "step_1": {
                "title": "Install Twilio Package",
                "command": "pip install twilio",
                "description": "Install the Twilio Python SDK"
            },
            "step_2": {
                "title": "Create Twilio Account",
                "url": "https://www.twilio.com/try-twilio",
                "description": "Sign up for a Twilio account to get your credentials"
            },
            "step_3": {
                "title": "Get Twilio Credentials",
                "instructions": [
                    "Log in to your Twilio Console",
                    "Find your Account SID and Auth Token",
                    "Purchase a phone number or use trial number"
                ]
            },
            "step_4": {
                "title": "Configure Environment Variables",
                "variables": {
                    "TWILIO_ACCOUNT_SID": "Your Twilio Account SID",
                    "TWILIO_AUTH_TOKEN": "Your Twilio Auth Token",
                    "TWILIO_PHONE_NUMBER": "Your Twilio phone number (with country code)"
                },
                "example_env_file": """# Add these to your backend/.env file
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_auth_token_here
TWILIO_PHONE_NUMBER=+1234567890"""
            },
            "step_5": {
                "title": "Test Configuration",
                "endpoint": "/admin/test_sms",
                "description": "Use the test SMS endpoint to verify everything is working"
            }
        },
        "troubleshooting": {
            "common_issues": [
                {
                    "issue": "SMS service not available",
                    "solution": "Install Twilio package: pip install twilio"
                },
                {
                    "issue": "Authentication failed",
                    "solution": "Check your Account SID and Auth Token are correct"
                },
                {
                    "issue": "Invalid phone number",
                    "solution": "Ensure phone numbers include country code (e.g., +91 for India)"
                },
                {
                    "issue": "Trial account limitations",
                    "solution": "Twilio trial accounts can only send to verified numbers"
                }
            ]
        },
        "features": {
            "automatic_notifications": "Status updates are automatically sent via SMS",
            "phone_number_formatting": "Indian phone numbers are automatically formatted",
            "fallback_behavior": "SMS failures fallback to email notifications",
            "logging": "All SMS attempts are logged for monitoring"
        }
    }

@app.get("/admin/debug_env")
def debug_environment_variables():
    """
    Debug endpoint to check environment variable loading
    (Remove this in production for security)
    """
    try:
        twilio_sid = os.environ.get('TWILIO_ACCOUNT_SID', 'NOT_FOUND')
        twilio_token = os.environ.get('TWILIO_AUTH_TOKEN', 'NOT_FOUND')
        twilio_phone = os.environ.get('TWILIO_PHONE_NUMBER', 'NOT_FOUND')
        
        return {
            "success": True,
            "environment_check": {
                "twilio_account_sid": f"{twilio_sid[:8]}...{twilio_sid[-4:]}" if twilio_sid != 'NOT_FOUND' and len(twilio_sid) > 12 else twilio_sid,
                "twilio_auth_token_length": len(twilio_token) if twilio_token != 'NOT_FOUND' else 0,
                "twilio_phone_number": twilio_phone,
                "env_file_path": "backend/.env",
                "working_directory": os.getcwd()
            },
            "note": "This is a debug endpoint - remove in production"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
