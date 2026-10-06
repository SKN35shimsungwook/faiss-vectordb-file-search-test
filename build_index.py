"""도로교통법 PDF -> 조문 단위 청크 -> 임베딩 -> FAISS 인덱스 저장"""
import json
import re
from pathlib import Path

import faiss
import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).parent
PDF_PATH = BASE / "data" / "도로교통법.pdf"
INDEX_PATH = BASE / "index" / "road_traffic.faiss"
META_PATH = BASE / "index" / "chunks.json"
MODEL_NAME = "jhgan/ko-sroberta-multitask"  # 한국어 문장 임베딩 모델 (768차원)
MAX_CHARS = 180  # 모델 최대 입력 128토큰(한글 약 200자) 안에 들어가도록 분할

# 머리말/꼬리말(법제처, 국가법령정보센터, 쪽번호) 제거용
NOISE = re.compile(r"^(도로교통법|법제처\s+\d+\s+국가법령정보센터)$")
# "제12조(제목)" / "제12조의2(제목)" 로 시작하는 줄 = 조문 시작
ARTICLE_START = re.compile(r"^(제\d+조(?:의\d+)?)\(([^)]+)\)")
CHAPTER = re.compile(r"^제\d+장(?:의\d+)?\s+\S+")


def extract_lines(pdf_path):
    lines = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            for line in (page.extract_text() or "").splitlines():
                line = line.strip()
                if line and not NOISE.match(line):
                    lines.append((page_no, line))
    return lines


def split_articles(lines):
    articles, cur, chapter = [], None, ""
    for page_no, line in lines:
        if CHAPTER.match(line):
            chapter = line
            continue
        m = ARTICLE_START.match(line)
        if m:
            if cur:
                articles.append(cur)
            cur = {"article": m.group(1), "title": m.group(2), "chapter": chapter,
                   "page": page_no, "text": line}
        elif cur:
            cur["text"] += "\n" + line
    if cur:
        articles.append(cur)
    return articles


def to_chunks(articles):
    chunks = []
    for a in articles:
        header = f"[{a['article']}({a['title']})] "
        body = a["text"]
        # 긴 조문은 줄 단위로 MAX_CHARS 넘지 않게 묶어서 분할, 조문명은 모든 조각에 붙임
        parts, buf = [], ""
        for line in body.splitlines():
            if buf and len(buf) + len(line) > MAX_CHARS:
                parts.append(buf)
                buf = ""
            buf += line + "\n"
        if buf:
            parts.append(buf)
        for i, p in enumerate(parts):
            text = p.strip() if i == 0 else header + p.strip()
            chunks.append({**{k: a[k] for k in ("article", "title", "chapter", "page")},
                           "part": i + 1, "n_parts": len(parts), "text": text})
    return chunks


def main():
    lines = extract_lines(PDF_PATH)
    articles = split_articles(lines)
    chunks = to_chunks(articles)
    print(f"추출 줄 수: {len(lines)} / 조문 수: {len(articles)} / 청크 수: {len(chunks)}")

    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode([c["text"] for c in chunks], batch_size=32,
                       show_progress_bar=True, normalize_embeddings=True)
    emb = np.asarray(emb, dtype="float32")

    # 정규화 벡터 + 내적(IP) = 코사인 유사도
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    INDEX_PATH.parent.mkdir(exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    META_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"저장 완료: {INDEX_PATH.name} (벡터 {index.ntotal}개, {emb.shape[1]}차원)")


if __name__ == "__main__":
    main()
