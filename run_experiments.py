"""여러 질문으로 검색을 돌려 results.md 로 저장"""
from pathlib import Path

from search import Searcher

# (질문, 정답으로 기대하는 조문)
QUERIES = [
    ("음주운전으로 보는 혈중알코올농도 기준은?", "제44조"),
    ("술 마시고 운전하면 어떤 처벌을 받나요?", "제148조의2"),
    ("어린이 보호구역에서 지켜야 할 것", "제12조"),
    ("운전 중에 휴대폰 사용해도 되나요?", "제49조"),
    ("고속도로에서 갓길로 달려도 되나요?", "제60조"),
    ("교통사고를 내면 운전자가 해야 할 조치", "제54조"),
    ("운전면허가 취소되는 경우", "제93조"),
    ("자전거 탈 때 헬멧을 써야 하나요?", "제50조"),
    ("주차가 금지된 장소는 어디인가?", "제33조"),
    ("긴급자동차에게 길을 양보해야 하는 의무", "제29조"),
]
K = 3


def main():
    s = Searcher()
    out = [f"# 검색 실험 결과 (top-{K})\n"]
    summary = ["| # | 질문 | 기대 조문 | 1위 결과 | 점수 | 기대 조문 순위 |", "|---|---|---|---|---|---|"]
    hit1 = hit3 = 0
    for qi, (q, expected) in enumerate(QUERIES, start=1):
        results = s.search(q, K)
        ranks = [r for r, x in enumerate(results, start=1) if x["article"] == expected]
        rank = ranks[0] if ranks else None
        hit1 += rank == 1
        hit3 += rank is not None
        top = results[0]
        summary.append(f"| {qi} | {q} | {expected} | {top['article']}({top['title']}) | "
                       f"{top['score']:.3f} | {rank or '없음'} |")
        out.append(f"\n## Q{qi}. {q}\n\n기대 조문: **{expected}**\n")
        for r, x in enumerate(results, start=1):
            mark = " ✅" if x["article"] == expected else ""
            text = x["text"].replace("\n", " ")
            out.append(f"{r}. `{x['score']:.4f}` **{x['article']}({x['title']})** p.{x['page']}{mark}\n"
                       f"   > {text[:200]}\n")
    n = len(QUERIES)
    summary.append(f"\n**Hit@1 = {hit1}/{n}, Hit@{K} = {hit3}/{n}**\n")
    Path(__file__).with_name("results.md").write_text(
        out[0] + "\n## 요약\n\n" + "\n".join(summary) + "\n" + "".join(out[1:]), encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
