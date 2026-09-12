import os
import re
import json
import math
import urllib.request
import urllib.error
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")

# ---------------------------------------------------------------------------
# 정직성 노트 (제출 전 필독)
#
# 이 파일은 세 개의 분석 엔진을 우선순위대로 시도한다.
#   1) GEMINI (engine="llm_gemini"): GEMINI_API_KEY가 있으면 최우선 시도.
#      Google AI Studio 무료 티어(gemini-*-flash-lite 계열)는 카드 등록 없이
#      발급 가능해서 기본값으로 이 경로를 권장한다. 비용 걱정 없이 "진짜 LLM
#      호출"을 데모에서 보여줄 수 있는 경로.
#   2) OPENAI (engine="llm_openai"): OPENAI_API_KEY가 있고 Gemini가 없거나
#      실패했을 때 시도. 카드/결제 등록이 필요한 유료 경로라는 걸 알고 쓸 것.
#   3) RULE_BASED_FALLBACK (engine="rule_based_fallback"): 키가 하나도 없거나
#      둘 다 실패했을 때만 쓰는 키워드 기반 보조 로직. 데모 안정성을 위한
#      폴백이지 "AI 분석"이 아니다.
#
# 응답의 "engine" 필드가 실제로 어떤 경로를 탔는지 그대로 보여준다.
# 신청서/발표자료에는 이 필드값을 근거로만 "AI로 분석했다"고 써야 한다.
# 측정하지 않은 정확도(Precision/F1 등)는 여기서도, 어디에서도 하드코딩하지 않는다 —
# 실제 평가셋을 만들기 전까지는 "목표치"라고만 표기할 것.
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Life Fit Engine (Prototype)",
    description="Wanted AI Championship 2026 - Multidimensional Career-Lifestyle Alignment Engine (MVP/Prototype)",
    version="0.3.0-prototype"
)

# CORS: allow_origins="*" 와 allow_credentials=True는 스펙상 동시에 쓸 수 없다
# (브라우저가 거부). 이 데모는 쿠키/인증정보를 안 쓰므로 credentials는 끈다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

DIMENSION_KEYS = [
    "autonomy",             # 자율성 / 재량권
    "deep_work",            # 집중 몰입 시간 확보
    "social_load",          # 대외/팀 소통 에너지 부하
    "pace_predictability",  # 일정 및 루틴의 예측 가능성
    "async_collaboration",  # 비동기/문서 중심 협업 (vs 실시간 미팅)
    "boundary_integrity"    # 업무와 휴식의 명확한 경계
]

# ----------------- 데이터 모델 -----------------
class LogInput(BaseModel):
    text: str

class UserValidationInput(BaseModel):
    user_vector: Dict[str, float]
    weights: Optional[Dict[str, float]] = None

# ----------------- 채용공고 6축 데이터베이스 (데모용 샘플 3건, 실 데이터 아님) -----------------
# 회사명은 전부 가상 기업입니다. 실존 기업의 실제 조직문화를 조사/반영한 것이 아니므로
# 특정 기업을 연상시키는 실명은 쓰지 않습니다 (근거 없는 문화 점수를 실명 기업에
# 붙이는 것 자체가 신뢰성/평판 리스크이기 때문).
JD_DATABASE = [
    {
        "id": "wanted-jd-01",
        "company": "페이플로 (가상 핀테크사)",
        "title": "Senior Backend Developer (Core)",
        "summary": "비동기 문서 중심 협업, 주 2회 하이브리드, 깊은 몰입 시간 보장",
        "vector": {
            "autonomy": 0.90, "deep_work": 0.95, "social_load": 0.25,
            "pace_predictability": 0.65, "async_collaboration": 0.90, "boundary_integrity": 0.70
        }
    },
    {
        "id": "wanted-jd-02",
        "company": "로컬허브 (가상 커머스사)",
        "title": "Community & CX Operations Specialist",
        "summary": "실시간 유저 피드백 대응, 일일 스탠드업 및 대면 싱크 위주",
        "vector": {
            "autonomy": 0.45, "deep_work": 0.30, "social_load": 0.90,
            "pace_predictability": 0.40, "async_collaboration": 0.25, "boundary_integrity": 0.45
        }
    },
    {
        "id": "wanted-jd-03",
        "company": "홈스타일랩 (가상 라이프스타일 플랫폼)",
        "title": "Growth Product Manager (PO)",
        "summary": "크로스 펑셔널 스쿼드 리딩, 빠른 가설 검증 및 다부서 의견 조율",
        "vector": {
            "autonomy": 0.85, "deep_work": 0.50, "social_load": 0.80,
            "pace_predictability": 0.55, "async_collaboration": 0.65, "boundary_integrity": 0.60
        }
    }
]

# ----------------- Stage 1: PII 마스킹 (실제 동작하는 최소 구현) -----------------
# 완전한 PII 탐지는 아니지만, "Zero-Retention/마스킹" 주장을 코드로 뒷받침하는
# 최소 정규식 기반 구현. 원본 텍스트는 마스킹 후 즉시 버려지고 응답/로그에 남기지 않는다.
_PII_PATTERNS = [
    (re.compile(r"\b01[016789]-?\d{3,4}-?\d{4}\b"), "[전화번호 마스킹됨]"),
    (re.compile(r"[\w\.-]+@[\w\.-]+\.\w+"), "[이메일 마스킹됨]"),
    (re.compile(r"\b\d{6}-?[1-4]\d{6}\b"), "[주민등록번호 형식 마스킹됨]"),
    (re.compile(r"[가-힣]{2,4}(?=\s?(이랑|이랑은|씨는|씨와|와 사귀|와 결혼|남편|아내|여자친구|남자친구))"), "[관계인 이름 마스킹됨]"),
]

def mask_pii(text: str) -> Dict[str, Any]:
    masked = text
    hit_count = 0
    for pattern, placeholder in _PII_PATTERNS:
        masked, n = pattern.subn(placeholder, masked)
        hit_count += n
    return {"masked_text": masked, "masked_count": hit_count}

# ----------------- Stage 2: 규칙 기반 폴백 파서 (LLM 미사용 시에만 동작) -----------------
def rule_based_parse(text: str) -> Dict[str, Any]:
    vec = {k: 0.50 for k in DIMENSION_KEYS}
    evidences = []
    signals = []

    if "[Google Calendar" in text:
        vec = {
            "autonomy": 0.40, "deep_work": 0.25, "social_load": 0.90,
            "pace_predictability": 0.45, "async_collaboration": 0.30, "boundary_integrity": 0.35
        }
        evidences.append({
            "id": 1, "category": "meeting_overload",
            "text": "주간 18.5시간 미팅 및 집중 블록 부족 (데모용 샘플 데이터)", "confidence": 0.60
        })
        signals.append({
            "id": "sig-cal", "dimension": "social_load",
            "hypothesis": "회의 부하로 인한 몰입 시간 부족 가능성", "confidence": 0.60
        })
        return {"user_vector": vec, "evidence": evidences, "signals": signals}

    if "[Slack Activity" in text:
        vec = {
            "autonomy": 0.75, "deep_work": 0.60, "social_load": 0.70,
            "pace_predictability": 0.50, "async_collaboration": 0.85, "boundary_integrity": 0.30
        }
        evidences.append({
            "id": 1, "category": "boundary_breach",
            "text": "야간 시간대 메시지 응답 빈도 높음 (데모용 샘플 데이터)", "confidence": 0.60
        })
        signals.append({
            "id": "sig-slk", "dimension": "boundary_integrity",
            "hypothesis": "퇴근 후 경계 보호가 필요할 가능성", "confidence": 0.60
        })
        return {"user_vector": vec, "evidence": evidences, "signals": signals}

    if any(w in text for w in ["코딩", "혼자", "조용", "집중", "노트북", "문서"]):
        vec["deep_work"] = 0.85
        vec["autonomy"] = 0.80
        vec["social_load"] = 0.30
        vec["async_collaboration"] = 0.80
        vec["boundary_integrity"] = 0.70
        evidences.append({
            "id": 1, "category": "deep_work",
            "text": "몰입/집중 관련 키워드 감지 (규칙 기반 추정)", "confidence": 0.55
        })
        signals.append({
            "id": "sig-1", "dimension": "deep_work",
            "hypothesis": "자율성과 몰입 시간이 보장되는 환경 선호 가능성", "confidence": 0.55
        })

    if any(w in text for w in ["미팅", "회의", "싱크", "설득", "피곤", "진이 다 빠"]):
        vec["social_load"] = max(vec["social_load"], 0.75)
        vec["boundary_integrity"] = min(vec["boundary_integrity"], 0.45)
        evidences.append({
            "id": 2, "category": "social_drain",
            "text": "회의/소통 관련 키워드 감지 (규칙 기반 추정)", "confidence": 0.55
        })
        signals.append({
            "id": "sig-2", "dimension": "social_load",
            "hypothesis": "실시간 소통 부하가 클 가능성", "confidence": 0.55
        })

    if any(w in text for w in ["예측 불가", "갑자기", "긴급", "로드맵", "분기 단위", "정돈된 일정"]):
        vec["pace_predictability"] = 0.80 if any(w in text for w in ["로드맵", "분기 단위", "정돈된 일정"]) else 0.25
        evidences.append({
            "id": 3, "category": "pace_predictability",
            "text": "일정 예측 가능성 관련 키워드 감지 (규칙 기반 추정)", "confidence": 0.55
        })
        signals.append({
            "id": "sig-3", "dimension": "pace_predictability",
            "hypothesis": "예측 가능한 로드맵 vs 돌발 업무에 대한 선호 시그널", "confidence": 0.55
        })

    if any(w in text for w in ["비동기", "슬랙", "스레드", "문서화", "메신저"]):
        vec["async_collaboration"] = max(vec["async_collaboration"], 0.80)
        evidences.append({
            "id": 4, "category": "async_collaboration",
            "text": "비동기 협업 관련 키워드 감지 (규칙 기반 추정)", "confidence": 0.55
        })
        signals.append({
            "id": "sig-4", "dimension": "async_collaboration",
            "hypothesis": "비동기/문서 중심 협업 선호 가능성", "confidence": 0.55
        })

    if any(w in text for w in ["퇴근 후", "야근", "칼퇴", "저녁", "주말에도", "알림 차단"]):
        vec["boundary_integrity"] = 0.30 if any(w in text for w in ["야근", "저녁", "주말에도"]) else 0.80
        evidences.append({
            "id": 5, "category": "boundary_integrity",
            "text": "일-삶 경계 관련 키워드 감지 (규칙 기반 추정)", "confidence": 0.55
        })
        signals.append({
            "id": "sig-5", "dimension": "boundary_integrity",
            "hypothesis": "퇴근 후 경계 보호 수준에 대한 시그널", "confidence": 0.55
        })

    if not evidences:
        evidences.append({"id": 1, "category": "general", "text": "키워드 매칭 실패 - 중립값 반환 (규칙 기반 한계)", "confidence": 0.30})
        signals.append({"id": "sig-0", "dimension": "autonomy", "hypothesis": "판단 근거 부족 - 슬라이더로 직접 조정 필요", "confidence": 0.30})

    return {"user_vector": vec, "evidence": evidences, "signals": signals}

# ----------------- Stage 2 방어 코드: LLM 응답 검증 (malformed evidence/signals로 인한 프론트 NaN% 렌더링 방지) -----------------
def _safe_float01(value: Any, default: float = 0.5) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(f) or math.isinf(f):
        return default
    return max(0.0, min(1.0, f))

def sanitize_llm_result(parsed: Dict[str, Any]) -> Dict[str, Any]:
    raw_vector = parsed.get("user_vector") if isinstance(parsed, dict) else None
    user_vector = {
        k: _safe_float01((raw_vector or {}).get(k), 0.5) for k in DIMENSION_KEYS
    }

    raw_evidence = parsed.get("evidence") if isinstance(parsed, dict) else None
    evidence = []
    if isinstance(raw_evidence, list):
        for i, item in enumerate(raw_evidence):
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            category = item.get("category")
            evidence.append({
                "id": item.get("id", i + 1),
                "category": category if isinstance(category, str) and category.strip() else "general",
                "text": text,
                "confidence": _safe_float01(item.get("confidence"), 0.5),
            })
    if not evidence:
        evidence.append({"id": 1, "category": "general", "text": "LLM 응답 형식 이상 - 중립값 반환", "confidence": 0.5})

    raw_signals = parsed.get("signals") if isinstance(parsed, dict) else None
    signals = []
    if isinstance(raw_signals, list):
        for i, item in enumerate(raw_signals):
            if not isinstance(item, dict):
                continue
            hypothesis = item.get("hypothesis")
            if not isinstance(hypothesis, str) or not hypothesis.strip():
                continue
            dimension = item.get("dimension")
            signals.append({
                "id": item.get("id", f"sig-{i + 1}"),
                "dimension": dimension if isinstance(dimension, str) and dimension.strip() else "autonomy",
                "hypothesis": hypothesis,
                "confidence": _safe_float01(item.get("confidence"), 0.5),
            })
    if not signals:
        signals.append({"id": "sig-0", "dimension": "autonomy", "hypothesis": "LLM 응답 형식 이상 - 슬라이더로 직접 조정 필요", "confidence": 0.5})

    return {"user_vector": user_vector, "evidence": evidence, "signals": signals}

_LLM_PROMPT_TEMPLATE = (
    "다음 라이프로그 텍스트를 읽고 아래 6개 축을 0.0~1.0으로 추정해줘: "
    + ", ".join(DIMENSION_KEYS)
    + ". 반드시 JSON으로만 답해: "
      '{{"user_vector": {{...}}, "evidence": [{{"id":1,"category":"...","text":"...","confidence":0.0}}], '
      '"signals": [{{"id":"sig-1","dimension":"...","hypothesis":"...","confidence":0.0}}]}}\n\n'
      "텍스트: {text}"
)

# ----------------- Stage 2: Gemini 파서 (GEMINI_API_KEY 있을 때 최우선 시도, 무료 티어) -----------------
def gemini_parse(text: str) -> Optional[Dict[str, Any]]:
    if not GEMINI_API_KEY:
        return None
    try:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
        )
        body = {
            "contents": [{"parts": [{"text": _LLM_PROMPT_TEMPLATE.format(text=text)}]}],
            "generationConfig": {"responseMimeType": "application/json"},
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(raw_text)
    except urllib.error.HTTPError as e:
        print(f"[gemini_parse] Gemini 호출 실패({e.code}), 다음 엔진으로 폴백: {e.read()[:300]}")
        return None
    except Exception as e:
        print(f"[gemini_parse] Gemini 호출 실패, 다음 엔진으로 폴백: {e}")
        return None

# ----------------- Stage 2: OpenAI 파서 (OPENAI_API_KEY 있을 때, 유료) -----------------
def openai_parse(text: str) -> Optional[Dict[str, Any]]:
    if not OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": _LLM_PROMPT_TEMPLATE.format(text=text)}],
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"[openai_parse] OpenAI 호출 실패, 다음 엔진으로 폴백: {e}")
        return None

# ----------------- API Endpoints -----------------
@app.get("/api/v3/health")
def health():
    return {
        "status": "ok",
        "version": "0.3.0-prototype",
        "gemini_enabled": bool(GEMINI_API_KEY),  # 카드 등록 없는 무료 경로
        "openai_enabled": bool(OPENAI_API_KEY),  # 카드 등록 필요한 유료 경로
    }

@app.post("/api/v3/analyze-pipeline")
def analyze(req: LogInput):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text가 비어 있습니다.")
    if len(text) > 5000:
        text = text[:5000]  # 과도하게 긴 입력으로 인한 LLM 비용/지연 방지

    pii_result = mask_pii(text)
    cleaned_text = pii_result["masked_text"]
    masked_count = pii_result["masked_count"]

    try:
        # 우선순위: Gemini(무료, 카드 불필요) -> OpenAI(유료) -> 규칙 기반 폴백
        gemini_result = gemini_parse(cleaned_text)
        if gemini_result is not None:
            engine = "llm_gemini"
            parsed = sanitize_llm_result(gemini_result)
        else:
            openai_result = openai_parse(cleaned_text)
            if openai_result is not None:
                engine = "llm_openai"
                parsed = sanitize_llm_result(openai_result)
            else:
                engine = "rule_based_fallback"
                parsed = rule_based_parse(cleaned_text)
    except Exception as e:
        # 데모 중 500으로 죽는 것보다, 무슨 일이 있었는지 밝히고 안전하게 폴백
        print(f"[analyze] 처리 중 오류, 규칙 기반으로 폴백: {e}")
        engine = "rule_based_fallback"
        parsed = rule_based_parse(cleaned_text)

    return {
        "status": "success",
        "engine": engine,  # "llm_gemini"/"llm_openai"가 아니면 "AI가 분석했다"고 쓰지 말 것
        "stage1_privacy": {
            "masked_text": cleaned_text,
            "masked_count": masked_count,
            "note": "정규식 기반 최소 마스킹. 완전한 PII 탐지를 보장하지 않음."
        },
        "stage2_inference": parsed
    }

@app.post("/api/v3/match-jds")
def match(val: UserValidationInput):
    if not val.user_vector:
        raise HTTPException(status_code=400, detail="user_vector가 비어 있습니다.")
    # 슬라이더 조작 실수나 손상된 요청으로 0~1 범위를 벗어난 값이 와도 죽지 않게 클램프
    u_vec = {k: max(0.0, min(1.0, float(val.user_vector.get(k, 0.5)))) for k in DIMENSION_KEYS}
    results = []

    for jd in JD_DATABASE:
        j_vec = jd["vector"]

        dot = sum(u_vec.get(k, 0.5) * j_vec.get(k, 0.5) for k in DIMENSION_KEYS)
        mag_u = math.sqrt(sum(u_vec.get(k, 0.5) ** 2 for k in DIMENSION_KEYS))
        mag_j = math.sqrt(sum(j_vec.get(k, 0.5) ** 2 for k in DIMENSION_KEYS))
        cos_sim = dot / (mag_u * mag_j) if (mag_u * mag_j) > 0 else 0.5

        diff = sum(abs(u_vec.get(k, 0.5) - j_vec.get(k, 0.5)) for k in DIMENSION_KEYS) / 6.0
        dist_score = max(0.0, 1.0 - diff)

        score = (cos_sim * 0.40 + dist_score * 0.60) * 100

        penalty_msg = None
        if u_vec.get("social_load", 0.5) > 0.75 and j_vec.get("social_load", 0.5) > 0.80:
            score *= 0.88
            penalty_msg = "회의 과부하 상태에서 추가 소통 부하로 인한 번아웃 위험 (휴리스틱 경고, 실측 아님)"

        final_score = round(max(0.0, min(100.0, score)), 1)

        # NOTE: 아래는 실측 데이터로 피팅한 "예측 모델"이 아니라, 데모 설명용 예시
        # 스코어링 공식이다. 실제 근속 데이터가 없으므로 신청서에는 "예측 모델"이
        # 아니라 "일러스트레이티브 지표(예시 공식)"라고 표기할 것.
        illustrative_retention = round(min(98.0, max(42.0, 36.0 + final_score * 0.62)), 1)

        dimension_details = {}
        for dim in DIMENSION_KEYS:
            d_diff = abs(u_vec.get(dim, 0.5) - j_vec.get(dim, 0.5))
            dimension_details[dim] = {
                "sim": round((1.0 - d_diff) * 100),
                "status": "Optimal" if d_diff <= 0.20 else ("Warning" if d_diff >= 0.45 else "Moderate")
            }

        results.append({
            "id": jd["id"],
            "company": jd["company"],
            "title": jd["title"],
            "finalScore": final_score,
            "illustrativeRetentionRate": illustrative_retention,
            "culture": jd["summary"],
            "riskWarning": penalty_msg,
            "details": dimension_details,
            "vector": j_vec
        })

    results.sort(key=lambda x: x["finalScore"], reverse=True)
    return {"status": "success", "matches": results, "note": "illustrativeRetentionRate는 실측 근속 데이터로 검증되지 않은 예시 공식입니다."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
