# Unified Multi-Model SHAP+RAG(+Image) Experiment Guide

이 가이드는 내일 오전 전까지 빠르게 4개 generation model로 `SHAP+RAG`와 `SHAP+RAG+Image one-step`을 비교하기 위한 실행 절차입니다.

## 0. 이번 재구성에서 추가된 파일

```text
paper_experiment_suite/
  providers/
    base.py
    factory.py
    openai_provider.py
    gemini_provider.py
    openai_compatible_provider.py
    ollama_provider.py
  prompts/
    unified_report_prompt.py
    unified_geval_prompt.py
  config/
    model_matrix_fast.yaml
    unified_rubric.yaml
  scripts/
    run_unified_fast_experiment.sh
  paper_12_generate_multimodel_reports.py
  paper_13_evaluate_unified_geval.py
  requirements_unified_models.txt
```

기존 `paper_04`, `paper_08`, `paper_11`은 보존했습니다. 새 실험은 `paper_12`와 `paper_13`만 실행하면 됩니다.

## 1. Python 클라이언트 의존성 설치

```bash
pip install -r paper_experiment_suite/requirements_unified_models.txt
```

Qwen을 vLLM 또는 SGLang으로 직접 띄우는 서버 환경은 별도 설치가 필요합니다. 클라이언트 코드에서는 OpenAI-compatible endpoint만 호출합니다.

## 2. API key 설정

```bash
export OPENAI_API_KEY="YOUR_OPENAI_KEY"
export GEMINI_API_KEY="YOUR_GEMINI_KEY"
```

`OPENAI_API_KEY`는 OpenAI generation과 unified G-Eval judge에 사용됩니다. `GEMINI_API_KEY`는 Gemini generation에 사용됩니다.

## 3. Llama 3.2 Vision 설치 및 실행

Ollama 설치 후:

```bash
ollama pull llama3.2-vision
ollama serve
```

별도 터미널에서 서버가 뜬 상태로 두세요. 기본 주소는 `http://localhost:11434`입니다.

간단 확인:

```bash
python - <<'PY'
from ollama import chat
r = chat(model='llama3.2-vision', messages=[{'role': 'user', 'content': 'Say OK.'}])
print(r.message.content)
PY
```

## 4. Qwen3-VL-8B 실행

### 권장: vLLM OpenAI-compatible server

```bash
pip install vllm
vllm serve "Qwen/Qwen3-VL-8B-Instruct" --host 0.0.0.0 --port 8000
```

기본 config는 `http://localhost:8000/v1`을 호출합니다.

간단 확인:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"Qwen/Qwen3-VL-8B-Instruct",
    "messages":[{"role":"user","content":"Say OK."}],
    "max_tokens":20
  }'
```

### 대안: SGLang

```bash
pip install sglang
python3 -m sglang.launch_server \
  --model-path "Qwen/Qwen3-VL-8B-Instruct" \
  --host 0.0.0.0 \
  --port 30000
```

이 경우 `paper_experiment_suite/config/model_matrix_fast.yaml`에서 Qwen base_url을 아래처럼 바꾸세요.

```yaml
qwen3_vl_8b:
  provider: openai_compatible
  model: Qwen/Qwen3-VL-8B-Instruct
  base_url: http://localhost:30000/v1
  default_api_key: EMPTY
```

## 5. Smoke test

먼저 API/server 연결만 확인합니다.

```bash
python paper_experiment_suite/paper_12_generate_multimodel_reports.py \
  --base_reports_json paper_outputs/reports/reports_shap_rag_stratified.json \
  --visual_cases_jsonl paper_outputs/visual/visual_cases.jsonl \
  --visual_inspection_jsonl paper_outputs/visual/visual_inspection_v2.jsonl \
  --model_config paper_experiment_suite/config/model_matrix_fast.yaml \
  --models openai_mini gemini_flash qwen3_vl_8b llama32_vision_11b \
  --methods shap_rag shap_rag_image_1step \
  --max_cases 2 \
  --n_repeats 1 \
  --output_dir paper_outputs/reports_multimodel_smoke \
  --overwrite
```

API 호출 없이 prompt 생성만 확인하려면 `--prompt_only`를 붙이세요.

## 6. Overnight full run

시간이 중요하므로 1차는 `n_repeats=1`, `judge_scope=first_report`로 돌리는 것을 권장합니다.

```bash
MAX_CASES=0 N_REPEATS=1 JUDGE_SCOPE=first_report \
  bash paper_experiment_suite/scripts/run_unified_fast_experiment.sh
```

전체가 너무 오래 걸리면 `MAX_CASES=30` 또는 `MAX_CASES=60`으로 제한하세요.

## 7. Unified evaluation만 따로 실행

```bash
python paper_experiment_suite/paper_13_evaluate_unified_geval.py \
  --reports \
    shap_rag__openai_mini=paper_outputs/reports_multimodel/reports_shap_rag__openai_mini.json \
    shap_rag_image__openai_mini=paper_outputs/reports_multimodel/reports_shap_rag_image_1step__openai_mini.json \
    shap_rag__gemini_flash=paper_outputs/reports_multimodel/reports_shap_rag__gemini_flash.json \
    shap_rag_image__gemini_flash=paper_outputs/reports_multimodel/reports_shap_rag_image_1step__gemini_flash.json \
    shap_rag__qwen3_vl_8b=paper_outputs/reports_multimodel/reports_shap_rag__qwen3_vl_8b.json \
    shap_rag_image__qwen3_vl_8b=paper_outputs/reports_multimodel/reports_shap_rag_image_1step__qwen3_vl_8b.json \
    shap_rag__llama32_vision_11b=paper_outputs/reports_multimodel/reports_shap_rag__llama32_vision_11b.json \
    shap_rag_image__llama32_vision_11b=paper_outputs/reports_multimodel/reports_shap_rag_image_1step__llama32_vision_11b.json \
  --visual_inspection_jsonl paper_outputs/visual/visual_inspection_v2.jsonl \
  --output_json paper_outputs/reports_multimodel/eval_unified_geval_4models.json \
  --run_geval \
  --geval_model gpt-4o \
  --judge_scope first_report \
  --resume
```

## 8. 출력 파일

Generation 결과:

```text
paper_outputs/reports_multimodel/reports_shap_rag__openai_mini.json
paper_outputs/reports_multimodel/reports_shap_rag_image_1step__openai_mini.json
...
```

Evaluation 결과:

```text
paper_outputs/reports_multimodel/eval_unified_geval_4models.json
```

`eval_unified_geval_4models.json` 안에는 method/model별 summary와 LaTeX table string이 포함됩니다.

## 9. 빠른 문제 해결

- `Missing OpenAI API key`: `export OPENAI_API_KEY=...` 확인
- `Missing Gemini API key`: `export GEMINI_API_KEY=...` 확인
- Qwen connection error: `curl http://localhost:8000/v1/models` 또는 server log 확인
- Ollama connection error: `ollama serve` 실행 여부 확인
- image missing: `--project_root`를 프로젝트 루트로 명시하거나 `paper_outputs/visual/images` 경로 존재 확인
- 시간이 부족함: `MAX_CASES=30`, `N_REPEATS=1`, `JUDGE_SCOPE=first_report` 유지
