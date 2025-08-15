import json
import re
from openai import OpenAI, APIError
from typing import Dict, Any, Optional
from .config import Config

class ResumeParser:
    def __init__(self):
        try:
            self.client = OpenAI(api_key=Config.get_api_key())
            self.model = Config.MODEL_NAME
            self.max_completion_tokens = Config.MAX_TOKENS
            self.temperature = Config.TEMPERATURE
        except ValueError as e:
            raise RuntimeError(f"OpenAI client initialization failed: {str(e)}")

    def parse_resume(self, text: str) -> Dict[str, Any]:
        """Main method to parse resume text"""
        try:
            # Validate input
            if not text or len(text.strip()) < 50:
                return {
                    "error": "Resume text too short",
                    "min_length": 50,
                    "received": len(text.strip()) if text else 0
                }
            # Get AI response
            response = self._get_ai_response(text)
            if "error" in response:
                return response

            # Parse and validate
            parsed_data = self._parse_and_validate(response, text)
            if "error" in parsed_data:
                return parsed_data

            return parsed_data

        except Exception as e:
            return {
                "error": f"Unexpected error: {str(e)}",
                "type": type(e).__name__,
                "suggestion": "Try again or contact support"
            }

    def _get_ai_response(self, text: str) -> Dict[str, Any]:
        """Get response from OpenAI API"""
        try:
            # Truncate input text to fit within token limits
            max_prompt_tokens = 3000  # Adjust based on model's token limit
            truncated_text = text[:max_prompt_tokens]

            # Log the truncated text and parameters for debugging
            prompt = self._create_prompt(truncated_text)
            print("Truncated Prompt Sent to OpenAI API:", prompt)

            # Dynamically calculate max_completion_tokens
            max_completion_tokens = 4096 - len(prompt) // 4  # Approximate token count

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                temperature=1,
                max_completion_tokens=max_completion_tokens
            )

            # Log the full response for debugging
            print("OpenAI API Response:", response)

            if not response.choices or not response.choices[0].message.content:
                return {
                    "error": "Empty response from OpenAI API",
                    "content_sample": None
                }

            return {"content": response.choices[0].message.content}

        except APIError as e:
            return {
                "error": f"OpenAI API error: {str(e)}",
                "code": e.code if hasattr(e, 'code') else None
            }

        except Exception as e:
            return {
                "error": f"Unexpected error: {str(e)}",
                "type": type(e).__name__
            }

    def _parse_and_validate(self, response: Dict[str, Any], original_text: str) -> Dict[str, Any]:
        """Parse and validate the AI response"""
        try:
            # Check if the content is empty
            if not response.get("content"):
                return {
                    "error": "Empty response from OpenAI API",
                    "content_sample": None
                }

            content = self._clean_json_response(response["content"])
            result = json.loads(content)

            # Validate minimum required fields
            if not result.get("name") and not result.get("email"):
                # Fallback to direct extraction
                result["name"] = self._extract_name_from_text(original_text) or "Not Found"
                result["email"] = self._extract_email_from_text(original_text) or "Not Found"

                if result["name"] == "Not Found" and result["email"] == "Not Found":
                    return {
                        "error": "Critical fields missing",
                        "missing_fields": ["name", "email"],
                        "raw_text_sample": original_text[:200] + "..." if original_text else None
                    }

            # Validate and clean skills
            result = self._validate_skills(result, original_text)

            # Clean empty values
            for key in list(result.keys()):
                if result[key] in [None, "", "Not Found", []]:
                    del result[key]

            return result

        except json.JSONDecodeError as e:
            return {
                "error": f"Invalid JSON response: {str(e)}",
                "content_sample": response["content"][:200] + "..." if "content" in response and response["content"] else None
            }


    def _create_prompt(self, text: str) -> str:
        """Create a simplified prompt for OpenAI"""
        return f"""
        Extract the following information from the resume:
        - Name
        - Email
        - Phone
        - LinkedIn URL
        - GitHub URL
        - Skills
        - Work Experience (Company, Position, Duration, Responsibilities)
        - Education (Institution, Degree, Field, Year)
        - Certifications (Name, Issuer, Date)
        - Projects (Name, Description, URL)

        Return the information in JSON format. If any field is missing, use null.

        Resume Content:
        {text}
        """
    def _clean_json_response(self, text: str) -> str:
        """Clean the JSON response from OpenAI"""
        text = re.sub(r'```json|```', '', text)
        # Fix common JSON issues
        text = re.sub(r',\s*([}\]])', r'\1', text)  # Trailing commas
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)  # Control chars
        return text.strip()

    def _extract_email_from_text(self, text: str) -> Optional[str]:
        """Fallback email extraction"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        match = re.search(email_pattern, text)
        return match.group(0) if match else None

    def _extract_name_from_text(self, text: str) -> Optional[str]:
        """Fallback name extraction"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        for line in lines[:5]:  # Check first 5 non-empty lines
            if re.match(r'^[A-Z][a-z]+ [A-Z][a-z]+$', line):  # Basic name pattern
                return line
        return None

    def _validate_skills(self, result: Dict[str, Any], original_text: str) -> Dict[str, Any]:
        """Validate and enhance skills extraction"""
        # Initialize skills list if not present
        if "skills" not in result:
            result["skills"] = []

        # Clean the skills list
        skills = result["skills"]

        # Remove any example values that might have slipped through
        example_skills = {"python", "sql", "example", "sample", "test"}
        cleaned_skills = [
            skill for skill in skills
            if (isinstance(skill, str) and
                skill.lower() not in example_skills and
                len(skill.strip()) > 1)
        ]

        # If we don't have enough skills, try extracting from text directly
        if len(cleaned_skills) < 3:  # Threshold for considering extraction insufficient
            extracted_skills = self._extract_skills_from_text(original_text)
            # Merge without duplicates
            existing_skills_lower = {s.lower() for s in cleaned_skills}
            for skill in extracted_skills:
                if skill.lower() not in existing_skills_lower:
                    cleaned_skills.append(skill)

        # Update the result with cleaned skills
        result["skills"] = sorted(list(set(cleaned_skills)))  # Remove duplicates and sort

        return result

    def _extract_skills_from_text(self, text: str) -> list[str]:
        """Fallback method to extract skills directly from resume text"""
        # Common skill section patterns
        skill_section_patterns = [
            r'(?i)(?:skills|technical skills|technologies|competencies)[:\s-]+\s*(.+?)(?:\n\s*\n|\n\s*\w|$)',
            r'(?i)(?:proficient in|expertise in|strong knowledge of)[:\s-]+\s*(.+?)(?:\n\s*\n|\n\s*\w|$)'
        ]

        found_skills = set()

        # Try to find skill sections first
        for pattern in skill_section_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                skills_section = match.group(1)
                # Split by various delimiters
                skills = re.split(r'[,/•\-•·]|\n-|\n•', skills_section)
                for skill in skills:
                    skill = re.sub(r'\([^)]*\)', '', skill)  # Remove parentheses content
                    skill = skill.strip()
                    if 2 < len(skill) < 50 and not skill.isnumeric():
                        found_skills.add(skill)

        # If no skill section found, look for skills throughout the text
        if not found_skills:
            common_skills = [
                'python', 'java', 'c\+\+', 'c#', 'javascript', 'typescript',
                'html', 'css', 'react', 'angular', 'vue', 'node\.js', 'django',
                'flask', 'spring', 'sql', 'mysql', 'postgresql', 'mongodb',
                'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'git',
                'machine learning', 'data analysis', 'tensorflow', 'pytorch'
            ]

            for skill_pattern in common_skills:
                if re.search(rf'\b{skill_pattern}\b', text, re.I):
                    found_skills.add(skill_pattern.replace('\\', '').title())

        return sorted(found_skills)