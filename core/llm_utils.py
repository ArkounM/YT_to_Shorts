import os
import json
import re
import google.generativeai as genai
from typing import List, Dict, Any, Optional

# --- Base LLM Provider Class ---
class LLMProvider:
    """Base class for LLM providers."""
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key
        self.model_name = model
        self.client = self._initialize_client()

    def _initialize_client(self):
        """Initialize the API client for the specific provider."""
        raise NotImplementedError("Subclasses must implement _initialize_client.")

    def generate_text(self, prompt: str, **kwargs) -> Optional[str]:
        """Generate text using the LLM."""
        raise NotImplementedError("Subclasses must implement generate_text.")

    def _parse_json_from_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Attempts to parse JSON from a string, handling cases where JSON is embedded."""
        try:
            # Try direct parsing first
            return json.loads(response_text)
        except json.JSONDecodeError:
            # If direct parsing fails, try to find JSON within Markdown code blocks or raw text
            match = re.search(r'```json\s*(\{[\s\S]*?\})\s*```|(\{[\s\S]*\})', response_text, re.DOTALL)
            if match:
                # Prioritize the first group (json block), then the second (raw json)
                json_str = match.group(1) or match.group(2)
                if json_str:
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError as e:
                        print(f"Failed to parse extracted JSON: {e}")
                        return None
            print("No valid JSON found in the LLM response.")
            return None

    def _format_time_for_llm(self, seconds: float) -> str:
        """Format seconds to HH:MM:SS for LLM prompts if needed."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

# --- Google Gemini Provider ---
class GeminiProvider(LLMProvider):
    """LLM Provider for Google Gemini."""
    DEFAULT_MODEL = "gemini-1.5-flash"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key not provided or found in GEMINI_API_KEY environment variable.")
        super().__init__(api_key=api_key, model=model or self.DEFAULT_MODEL)
        print(f"GeminiProvider initialized with model: {self.model_name}")

    def _initialize_client(self):
        try:
            genai.configure(api_key=self.api_key)
            return genai.GenerativeModel(self.model_name)
        except Exception as e:
            print(f"Failed to initialize Gemini API client: {e}")
            raise ConnectionError(f"Failed to initialize Gemini API client: {e}")

    def generate_text(self, prompt: str, **kwargs) -> Optional[str]:
        """Generate text using the Gemini API."""
        print(f"Sending prompt to Gemini model {self.model_name}...")
        try:
            response = self.client.generate_content(prompt, **kwargs)
            # print(f"Gemini raw response: {response}") # For debugging
            if response.parts:
                return response.text
            elif response.prompt_feedback and response.prompt_feedback.block_reason:
                print(f"Gemini content generation blocked: {response.prompt_feedback.block_reason}")
                if response.prompt_feedback.block_reason_message:
                     print(f"Message: {response.prompt_feedback.block_reason_message}")
                return None
            else:
                print("Gemini response did not contain expected content parts.")
                return None
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return None

    def generate_json(self, prompt: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Generate text and attempt to parse it as JSON."""
        raw_response = self.generate_text(prompt, **kwargs)
        if raw_response:
            return self._parse_json_from_response(raw_response)
        return None


# --- Placeholder for OpenRouterProvider ---
class OpenRouterProvider(LLMProvider):
    DEFAULT_MODEL = "gryphe/mythomax-l2-13b" # Example, user might specify

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, base_url: str = "https://openrouter.ai/api/v1"):
        api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OpenRouter API key not provided or found in OPENROUTER_API_KEY environment variable.")
        super().__init__(api_key=api_key, model=model or self.DEFAULT_MODEL)
        self.base_url = base_url
        self.client = None # Will be initialized by _initialize_client (e.g. using requests session)
        print(f"OpenRouterProvider initialized for model: {self.model_name}")

    def _initialize_client(self):
        # For OpenRouter, we typically use the 'requests' library.
        # No specific client object to return here, but we ensure API key is present.
        import requests # Import here to keep it optional if OpenRouter is not used
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost", # Optional, but good practice
            "X-Title": "AI Video Tools CLI" # Optional
        })
        return self.session # Return session for consistency, though not a "client" object

    def generate_text(self, prompt: str, **kwargs) -> Optional[str]:
        print(f"Sending prompt to OpenRouter model {self.model_name}...")
        payload = {
            "model": self.model_name,
            "prompt": prompt, # Some models use 'prompt', others 'messages'
            # "messages": [{"role": "user", "content": prompt}], # Alternative for chat models
            **kwargs # Allow passing other params like max_tokens, temperature
        }
        # Check if model prefers 'messages' format
        # This is a heuristic, OpenRouter docs should be consulted for specific models
        if any(m_type in self.model_name.lower() for m_type in ["gpt-", "claude-", "mistral", "llama"]):
            payload.pop("prompt", None)
            payload["messages"] = [{"role": "user", "content": prompt}]


        try:
            response = self.session.post(f"{self.base_url}/chat/completions", json=payload)
            response.raise_for_status() # Raise an exception for bad status codes
            data = response.json()
            if data.get("choices") and data["choices"][0].get("message"):
                return data["choices"][0]["message"]["content"]
            # Fallback for older completion models if any are still used this way
            elif data.get("choices") and data["choices"][0].get("text"):
                return data["choices"][0]["text"]
            else:
                print(f"OpenRouter response did not contain expected content: {data}")
                return None
        except Exception as e:
            print(f"Error calling OpenRouter API: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response content: {e.response.text}")
            return None

    def generate_json(self, prompt: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Generate text and attempt to parse it as JSON."""
        raw_response = self.generate_text(prompt, **kwargs)
        if raw_response:
            return self._parse_json_from_response(raw_response)
        return None

# --- Placeholder for OpenAIProvider ---
class OpenAIProvider(LLMProvider):
    DEFAULT_MODEL = "gpt-3.5-turbo"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided or found in OPENAI_API_KEY environment variable.")
        super().__init__(api_key=api_key, model=model or self.DEFAULT_MODEL)
        print(f"OpenAIProvider initialized with model: {self.model_name}")

    def _initialize_client(self):
        try:
            from openai import OpenAI # Import here
            return OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("OpenAI Python client not installed. Please run `pip install openai`.")
        except Exception as e:
            raise ConnectionError(f"Failed to initialize OpenAI client: {e}")

    def generate_text(self, prompt: str, **kwargs) -> Optional[str]:
        print(f"Sending prompt to OpenAI model {self.model_name}...")
        try:
            # Assuming chat completion endpoint as it's more common now
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            if completion.choices and completion.choices[0].message:
                return completion.choices[0].message.content.strip()
            return None
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return None

    def generate_json(self, prompt: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Generate text and attempt to parse it as JSON.
           OpenAI has a specific JSON mode that can be enabled.
        """
        print(f"Sending prompt to OpenAI model {self.model_name} for JSON output...")
        try:
            # Forcing JSON mode if available for the model
            # This might require specific model versions (e.g., gpt-3.5-turbo-1106 or later)
            response_format_arg = {}
            if "1106" in self.model_name or "0125" in self.model_name or "gpt-4" in self.model_name: # Heuristic for models supporting JSON mode
                 response_format_arg = {"response_format": {"type": "json_object"}}

            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                    {"role": "user", "content": prompt}
                ],
                **response_format_arg,
                **kwargs
            )
            if completion.choices and completion.choices[0].message:
                content = completion.choices[0].message.content.strip()
                return self._parse_json_from_response(content) # Use the robust parser
            return None
        except Exception as e:
            print(f"Error calling OpenAI API for JSON: {e}")
            return None

# --- Placeholder for DeepseekProvider ---
class DeepseekProvider(LLMProvider):
    DEFAULT_MODEL = "deepseek-chat" # Or "deepseek-coder"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, base_url: str ="https://api.deepseek.com"):
        api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("Deepseek API key not provided or found in DEEPSEEK_API_KEY environment variable.")
        super().__init__(api_key=api_key, model=model or self.DEFAULT_MODEL)
        self.base_url = base_url
        print(f"DeepseekProvider initialized with model: {self.model_name}")

    def _initialize_client(self):
        # Deepseek is OpenAI compatible, so we can use the OpenAI client
        try:
            from openai import OpenAI
            return OpenAI(api_key=self.api_key, base_url=self.base_url)
        except ImportError:
            raise ImportError("OpenAI Python client not installed (required for Deepseek). Please run `pip install openai`.")
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Deepseek client: {e}")

    def generate_text(self, prompt: str, **kwargs) -> Optional[str]:
        print(f"Sending prompt to Deepseek model {self.model_name}...")
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            if completion.choices and completion.choices[0].message:
                return completion.choices[0].message.content.strip()
            return None
        except Exception as e:
            print(f"Error calling Deepseek API: {e}")
            return None

    def generate_json(self, prompt: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Generate text and attempt to parse it as JSON."""
        # Deepseek API might support a JSON mode similar to OpenAI with newer models.
        # For now, use standard generation and parse.
        print(f"Sending prompt to Deepseek model {self.model_name} for JSON output...")
        try:
            # Check if model might support OpenAI-like JSON mode
            response_format_arg = {}
            # Example: if self.model_name has certain keywords and you know it supports it.
            # For now, not enabling by default unless Deepseek docs confirm it.

            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                    {"role": "user", "content": prompt}
                ],
                **response_format_arg, # Add if JSON mode is confirmed
                **kwargs
            )
            if completion.choices and completion.choices[0].message:
                content = completion.choices[0].message.content.strip()
                return self._parse_json_from_response(content)
            return None
        except Exception as e:
            print(f"Error calling Deepseek API for JSON: {e}")
            return None

# --- Factory function to get LLM provider ---
SUPPORTED_PROVIDERS = {
    "gemini": GeminiProvider,
    "openrouter": OpenRouterProvider,
    "openai": OpenAIProvider,
    "deepseek": DeepseekProvider,
}

def get_llm_provider(provider_name: str, api_key: Optional[str] = None, model: Optional[str] = None) -> LLMProvider:
    """
    Factory function to get an instance of an LLM provider.
    """
    provider_name = provider_name.lower()
    if provider_name not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported LLM provider: {provider_name}. Supported providers are: {list(SUPPORTED_PROVIDERS.keys())}")

    provider_class = SUPPORTED_PROVIDERS[provider_name]

    try:
        return provider_class(api_key=api_key, model=model)
    except ValueError as e: # Catch API key errors specifically
        print(f"Configuration error for {provider_name}: {e}")
        raise
    except ImportError as e: # Catch missing dependencies
        print(f"Import error for {provider_name}: {e}")
        raise
    except ConnectionError as e: # Catch client init errors
        print(f"Connection error for {provider_name}: {e}")
        raise

# Example usage (for testing this module directly)
if __name__ == '__main__':
    print("Testing LLM Utils...")
    # Note: These tests require API keys to be set as environment variables.
    # Test Gemini
    try:
        print("\n--- Testing Gemini ---")
        gemini_provider = get_llm_provider("gemini")
        # gemini_response = gemini_provider.generate_text("Tell me a short joke.")
        # print(f"Gemini Response: {gemini_response}")

        json_prompt_gemini = "Generate a JSON object with two keys: 'name' and 'type', describing a fruit."
        gemini_json_response = gemini_provider.generate_json(json_prompt_gemini)
        print(f"Gemini JSON Response: {gemini_json_response}")
        assert isinstance(gemini_json_response, dict)
        assert "name" in gemini_json_response
    except Exception as e:
        print(f"Gemini test failed (API key might be missing or invalid): {e}")

    # Test OpenRouter
    try:
        print("\n--- Testing OpenRouter ---")
        # You might need to specify a free model if your default requires credits and you don't have them
        openrouter_provider = get_llm_provider("openrouter", model="nousresearch/nous-capybara-7b-v1.9") # Example free model
        # openrouter_response = openrouter_provider.generate_text("What is the capital of France?")
        # print(f"OpenRouter Response: {openrouter_response}")

        json_prompt_or = "Output a JSON object with a 'city' and 'country' field for Paris."
        or_json_response = openrouter_provider.generate_json(json_prompt_or)
        print(f"OpenRouter JSON Response: {or_json_response}")
        if or_json_response: # Can be None if API call fails
            assert isinstance(or_json_response, dict)
            assert "city" in or_json_response
    except Exception as e:
        print(f"OpenRouter test failed (API key might be missing or invalid): {e}")

    # Test OpenAI
    try:
        print("\n--- Testing OpenAI ---")
        openai_provider = get_llm_provider("openai", model="gpt-3.5-turbo") # Ensure this model is accessible
        # openai_response = openai_provider.generate_text("Explain quantum computing in one sentence.")
        # print(f"OpenAI Response: {openai_response}")

        json_prompt_openai = "Provide a JSON object with 'item' and 'price' for a banana."
        openai_json_response = openai_provider.generate_json(json_prompt_openai) # Test JSON mode
        print(f"OpenAI JSON Response: {openai_json_response}")
        if openai_json_response:
            assert isinstance(openai_json_response, dict)
            assert "item" in openai_json_response
    except Exception as e:
        print(f"OpenAI test failed (API key might be missing or invalid): {e}")

    # Test Deepseek
    try:
        print("\n--- Testing Deepseek ---")
        deepseek_provider = get_llm_provider("deepseek", model="deepseek-chat")
        # deepseek_response = deepseek_provider.generate_text("Write a python function that sums two numbers.")
        # print(f"Deepseek Response:\n{deepseek_response}")

        json_prompt_ds = "Create a JSON with 'language' and 'version' for Python 3.9."
        ds_json_response = deepseek_provider.generate_json(json_prompt_ds)
        print(f"Deepseek JSON Response: {ds_json_response}")
        if ds_json_response:
            assert isinstance(ds_json_response, dict)
            assert "language" in ds_json_response

    except Exception as e:
        print(f"Deepseek test failed (API key might be missing or invalid): {e}")

    print("\nLLM Utils testing done.")
