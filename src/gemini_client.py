"""
Gemini API client for document category extraction
"""
import os
from typing import Optional, Dict, Any
import google.generativeai as genai
from pathlib import Path


class GeminiClient:
    """Client for interacting with Google's Gemini API"""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.0-flash-exp"):
        """
        Initialize Gemini client

        Args:
            api_key: Google API key. If None, reads from GOOGLE_API_KEY env variable
            model_name: Name of the Gemini model to use
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided or set in GOOGLE_API_KEY environment variable")

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)

    def extract_category(self, file_path: str) -> Dict[str, Any]:
        """
        Extract category information from a document using Gemini

        Args:
            file_path: Path to the document file

        Returns:
            Dictionary containing category information and extracted data
        """
        prompt = """
        Analyze this document and extract the following information:

        1. Document Category: Identify the type/category of this document (e.g., invoice, receipt, contract, resume, report, letter, form, etc.)
        2. Main Topic: What is the primary subject or topic of this document?
        3. Key Entities: List any important names, organizations, dates, or amounts mentioned
        4. Language: Identify the primary language(s) used in the document
        5. Confidence Level: Rate your confidence in the categorization (high/medium/low)

        Provide the response in a structured JSON format with the following keys:
        - category
        - main_topic
        - key_entities (as a list)
        - language
        - confidence
        - summary (brief 1-2 sentence summary)
        """

        # Upload the file to Gemini
        uploaded_file = genai.upload_file(file_path)

        # Generate response
        response = self.model.generate_content([prompt, uploaded_file])

        # Clean up the uploaded file
        genai.delete_file(uploaded_file.name)

        return {
            "status": "success",
            "result": response.text,
            "model": self.model.model_name
        }

    def extract_category_from_bytes(self, file_bytes: bytes, filename: str, mime_type: str) -> Dict[str, Any]:
        """
        Extract category information from file bytes

        Args:
            file_bytes: File content as bytes
            filename: Original filename
            mime_type: MIME type of the file

        Returns:
            Dictionary containing category information and extracted data
        """
        import tempfile

        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name

        try:
            result = self.extract_category(tmp_path)
            return result
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
