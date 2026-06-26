import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict
import re
from rag_knowledge_base import DEPARTMENT_KNOWLEDGE_BASE, search_departments_by_keywords

class RAGEnhancedClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        self.department_vectors = None
        self.department_texts = []
        self.department_names = []
        self._build_department_vectors()
    
    def _build_department_vectors(self):
        """Build TF-IDF vectors for all departments"""
        for dept_name, dept_info in DEPARTMENT_KNOWLEDGE_BASE.items():
            # Combine all text information for each department
            combined_text = []
            combined_text.append(dept_info.get("description", ""))
            combined_text.extend(dept_info.get("responsibilities", []))
            combined_text.extend(dept_info.get("example_grievances", []))
            combined_text.extend(dept_info.get("keywords", []))
            
            # Join all text
            dept_text = " ".join(combined_text)
            self.department_texts.append(dept_text)
            self.department_names.append(dept_name)
        
        # Create TF-IDF vectors
        self.department_vectors = self.vectorizer.fit_transform(self.department_texts)
    
    def get_relevant_context(self, grievance_text: str, top_k: int = 3) -> List[Dict]:
        """Get most relevant department contexts for a grievance"""
        # Transform the grievance text to vector
        grievance_vector = self.vectorizer.transform([grievance_text])
        
        # Calculate similarities
        similarities = cosine_similarity(grievance_vector, self.department_vectors)[0]
        
        # Get top-k most similar departments
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        relevant_contexts = []
        for idx in top_indices:
            dept_name = self.department_names[idx]
            similarity_score = similarities[idx]
            dept_info = DEPARTMENT_KNOWLEDGE_BASE[dept_name]
            
            relevant_contexts.append({
                "department": dept_name,
                "similarity_score": similarity_score,
                "description": dept_info.get("description", ""),
                "responsibilities": dept_info.get("responsibilities", []),
                "example_grievances": dept_info.get("example_grievances", [])[:3],  # Top 3 examples
                "keywords": dept_info.get("keywords", [])
            })
        
        return relevant_contexts
    
    def enhance_classification_prompt(self, grievance_text: str) -> str:
        """Create an enhanced prompt with RAG context"""
        # Get relevant context
        relevant_contexts = self.get_relevant_context(grievance_text, top_k=3)
        
        # Also get keyword-based matches
        keyword_matches = search_departments_by_keywords(grievance_text)[:2]
        
        # Build enhanced prompt
        prompt_parts = []
        
        prompt_parts.append(
            "You are an expert classifier for Tamil Nadu government departments. "
            "Use the provided context to make accurate classifications.\n"
        )
        
        # Add RAG context
        prompt_parts.append("RELEVANT DEPARTMENT CONTEXT:")
        for i, context in enumerate(relevant_contexts, 1):
            prompt_parts.append(f"\n{i}. {context['department']}:")
            prompt_parts.append(f"   Description: {context['description']}")
            prompt_parts.append(f"   Key Responsibilities: {', '.join(context['responsibilities'][:3])}")
            prompt_parts.append(f"   Similar Grievances: {', '.join(context['example_grievances'])}")
        
        # Add keyword matches if different from vector matches
        keyword_depts = [match[0] for match in keyword_matches]
        vector_depts = [ctx['department'] for ctx in relevant_contexts]
        
        additional_keyword_depts = [dept for dept in keyword_depts if dept not in vector_depts]
        if additional_keyword_depts:
            prompt_parts.append(f"\nADDITIONAL KEYWORD MATCHES: {', '.join(additional_keyword_depts)}")
        
        # Add department list
        prompt_parts.append("\nAVAILABLE DEPARTMENTS:")
        dept_list = list(DEPARTMENT_KNOWLEDGE_BASE.keys())
        prompt_parts.append(", ".join(dept_list))
        
        # Add classification instructions
        prompt_parts.append(
            f"\nGRIEVANCE TO CLASSIFY: '{grievance_text}'\n"
            "\nINSTRUCTIONS:"
            "\n1. Analyze the grievance against the provided context"
            "\n2. Match the grievance content with department responsibilities"
            "\n3. Consider similar example grievances from the context"
            "\n4. Return ONLY the exact department name from the available list"
            "\n5. If uncertain, choose the most relevant match from the context provided"
            "\n\nCLASSIFIED DEPARTMENT:"
        )
        
        return "\n".join(prompt_parts)
    
    def get_classification_confidence(self, grievance_text: str, classified_dept: str) -> float:
        """Calculate confidence score for the classification"""
        relevant_contexts = self.get_relevant_context(grievance_text, top_k=5)
        
        # Find the classified department in the relevant contexts
        for context in relevant_contexts:
            if context['department'] == classified_dept:
                return context['similarity_score']
        
        return 0.0  # Low confidence if department not in top matches
    
    def explain_classification(self, grievance_text: str, classified_dept: str) -> Dict:
        """Provide explanation for the classification"""
        relevant_contexts = self.get_relevant_context(grievance_text, top_k=3)
        keyword_matches = search_departments_by_keywords(grievance_text)
        
        explanation = {
            "classified_department": classified_dept,
            "confidence_score": self.get_classification_confidence(grievance_text, classified_dept),
            "top_matches": relevant_contexts,
            "keyword_matches": keyword_matches[:3],
            "reasoning": []
        }
        
        # Add reasoning
        for context in relevant_contexts:
            if context['department'] == classified_dept:
                explanation["reasoning"].append(
                    f"High semantic similarity ({context['similarity_score']:.3f}) with {classified_dept}"
                )
                break
        
        for match in keyword_matches:
            if match[0] == classified_dept:
                explanation["reasoning"].append(
                    f"Strong keyword match (score: {match[1]}) with {classified_dept}"
                )
                break
        
        return explanation

# Initialize the RAG classifier
rag_classifier = RAGEnhancedClassifier()