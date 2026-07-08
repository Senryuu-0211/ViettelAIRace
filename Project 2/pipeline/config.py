import os

from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

MAX_RETRIES = 3
REQUEST_TIMEOUT = 300
TEMPERATURE = 0.0

RXNAV_BASE_URL = "https://rxnav.nlm.nih.gov/REST"
ICD10_KB_PATH = os.getenv("ICD10_KB_PATH", "")

INPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "input")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

ASSERTION_VALID_TYPES = {"THUỐC", "CHẨN_ĐOÁN", "TRIỆU_CHỨNG"}
CANDIDATE_VALID_TYPES = {"THUỐC", "CHẨN_ĐOÁN"}
VALID_TYPES = {"THUỐC", "CHẨN_ĐOÁN", "TRIỆU_CHỨNG", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"}
VALID_ASSERTIONS = {"isNegated", "isHistorical", "isFamily"}
