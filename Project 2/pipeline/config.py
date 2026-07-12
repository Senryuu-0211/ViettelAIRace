import os

from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# --- Neo4j Config ---
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")


MAX_RETRIES = 3
REQUEST_TIMEOUT = 300
TEMPERATURE = 0.0
MAX_ASSERTION_BATCH = int(os.getenv("MAX_ASSERTION_BATCH", "8"))

RXNAV_BASE_URL = "https://rxnav.nlm.nih.gov/REST"

_PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))

# --- ICD-10 linker (hybrid retrieval trên KB TT06 local) ---
ICD_KB_PATH = os.getenv("ICD_KB_PATH", os.path.join(_PROJECT_DIR, "data", "icd10_tt06.json"))
EMBED_CACHE_DIR = os.path.join(_PROJECT_DIR, "data", "embed_cache")
# Model embedding — đổi biến này để test model nhẹ hơn (vd "paraphrase-multilingual", "nomic-embed-text")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3")
USE_EMBEDDING = os.getenv("USE_EMBEDDING", "1") not in ("0", "false", "False", "")
# Có embed cả tên mã lá không (chính xác hơn cho chẩn đoán cụ thể, nhưng ~18k vector, chậm lần đầu)
EMBED_LEAVES = os.getenv("EMBED_LEAVES", "1") not in ("0", "false", "False", "")
ICD_LEXICAL_WEIGHT = float(os.getenv("ICD_LEXICAL_WEIGHT", "0.5"))  # trọng số lexical trong fuse
ICD_MIN_SCORE = float(os.getenv("ICD_MIN_SCORE", "0.8"))           # ngưỡng chặn emit sai
ICD_TOP_K = int(os.getenv("ICD_TOP_K", "3"))

INPUT_DIR = os.path.join(_PROJECT_DIR, "input")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

ASSERTION_VALID_TYPES = {"THUỐC", "CHẨN_ĐOÁN", "TRIỆU_CHỨNG"}
CANDIDATE_VALID_TYPES = {"THUỐC", "CHẨN_ĐOÁN"}
VALID_TYPES = {"THUỐC", "CHẨN_ĐOÁN", "TRIỆU_CHỨNG", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"}
VALID_ASSERTIONS = {"isNegated", "isHistorical", "isFamily"}
