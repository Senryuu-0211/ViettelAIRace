"""
ICD-10 local lookup — minimal KB từ anchor set + BYT VN.
Khi có file KB đầy đủ thì thay thế module này.
"""

import json
import os
from typing import Optional

ICD10_KB: dict[str, list[str]] = {}

ANCHOR_ICD10 = {
    "bệnh trào ngược dạ dày-thực quản": ["K21.0", "K21.9"],
    "bệnh trào ngược dạ dày - thực quản": ["K21.0", "K21.9"],
    "trào ngược dạ dày thực quản": ["K21.0", "K21.9"],
    "tăng huyết áp": ["I10"],
    "tăng huyết áp vô căn": ["I10"],
    "đái tháo đường": ["E11.9"],
    "đái tháo đường type 2": ["E11.9"],
    "đái tháo đường type ii": ["E11.9"],
    "hen suyễn": ["J45.9"],
    "hen phế quản": ["J45.9"],
    "bệnh phổi tắc nghẽn mạn tính": ["J44.9"],
    "copd": ["J44.9"],
    "suy tim": ["I50.9"],
    "suy tim sung huyết": ["I50.0"],
    "nhồi máu cơ tim": ["I21.9"],
    "thiếu máu cơ tim": ["I25.9"],
    "đột quỵ": ["I64"],
    "tai biến mạch máu não": ["I64"],
    "xơ gan": ["K74.6"],
    "viêm gan": ["K75.9"],
    "suy thận": ["N19"],
    "suy thận mạn": ["N18.9"],
    "ung thư": ["C80.9"],
    "bệnh động mạch vành": ["I25.1"],
    "xơ vữa động mạch": ["I70.9"],
    "bệnh tim mạch do xơ vữa động mạch": ["I25.1"],
    "rối loạn lo âu": ["F41.9"],
    "trầm cảm": ["F32.9"],
    "loét dạ dày": ["K25.9"],
    "viêm phổi": ["J18.9"],
    "tăng lipid máu": ["E78.5"],
    "tăng lipid máu không đặc hiệu": ["E78.5"],
    "tăng lipid máu, không đặc hiệu": ["E78.5"],
    "béo phì": ["E66.9"],
    "thiếu máu": ["D64.9"],
    "hạ huyết áp": ["I95.9"],
    "hạ huyết áp không đặc hiệu": ["I95.9"],
    "hạ huyết áp, không đặc hiệu": ["I95.9"],
    "viêm phế quản": ["J40"],
    "viêm dạ dày": ["K29.7"],
    "tắc nghẽn đường mật": ["K83.1"],
    "giãn đường mật": ["K83.8"],
    "hội chứng não gan": ["K72.9"],
    "xơ gan do rượu": ["K70.3"],
    "khối u trực tràng": ["C20"],
    "u trực tràng": ["C20"],
    "u ác trực tràng": ["C20"],
    "u tuyến": ["D12.8"],
    "đau bụng": ["R10.9"],
}


def _init_kb():
    global ICD10_KB
    ICD10_KB.update(ANCHOR_ICD10)
    kb_path = os.getenv("ICD10_KB_PATH", "")
    if kb_path and os.path.exists(kb_path):
        with open(kb_path, "r", encoding="utf-8") as f:
            external = json.load(f)
            if isinstance(external, dict):
                ICD10_KB.update(external)


def lookup_diagnosis(diagnosis_text: str) -> list[str]:
    if not ICD10_KB:
        _init_kb()

    text = diagnosis_text.lower().strip()
    if text in ICD10_KB:
        return ICD10_KB[text]

    for key, codes in ICD10_KB.items():
        if key in text or text in key:
            return codes

    return []


def lookup_batch(diagnosis_texts: list[str]) -> dict[str, list[str]]:
    result = {}
    for text in diagnosis_texts:
        codes = lookup_diagnosis(text)
        if codes:
            result[text] = codes
    return result
