"""Pure extraction helpers for the approved collection scope; genre hints are unverified."""

import hashlib
import re

from korean_humanizer.corpus_pilot import excerpt_candidates, parse_article


def article_urls(markup: str) -> list[str]:
    return list(
        dict.fromkeys(
            "https://www.korea.kr/news/reporterView.do?newsId=" + value
            for value in re.findall(r"/news/reporterView\.do\?newsId=(\d+)", markup)
        )
    )


def genre_hint(title: str, paragraphs: list[str]) -> dict:
    """Triage only, never an accepted genre, semantic rating or training eligibility."""
    text = title + "\n" + "\n".join(paragraphs)
    patterns = {
        "experience_review": (
            r"체험|후기|다녀|가보니|가봤|둘러보|직접.{0,8}(?:찾|가|해|이용)|느꼈|인상적|맛보"
        ),
        "explanation_information": (
            r"신청 방법|이용 방법|지원 대상|알아보|제도|지원금|혜택|자격|사용법|절차"
        ),
        "opinion_column": (
            r"생각한다|생각합니다|필요하다|필요합니다|바란다|바랍니다|아쉬웠|아쉽다|문제점|개선해야"
        ),
        "practical_notice": r"접수 기간|신청 기간|신청 마감|모집 기간|제출 서류|신청하세요|접수처",
    }
    scores = {name: len(re.findall(pattern, text)) for name, pattern in patterns.items()}
    ordered = sorted(scores, key=lambda name: (-scores[name], name))
    top = ordered[0]
    label = top if scores[top] >= 2 and scores[top] > scores[ordered[1]] else "unclassified"
    return {"method": "rules_unverified", "label": label, "scores": scores}


def extract_candidates(markup: str, url: str, provenance: str):
    article = parse_article(markup, url, min_paragraph_chars=2)
    if int(article["published"][:4]) not in {2016, 2017, 2018}:
        raise ValueError("article year is outside the approved scale collection")
    candidates = excerpt_candidates(article, provenance, max_chars=600, min_chars=200, limit=4)
    sources = []
    for index, source in enumerate(candidates, 1):
        suffix = hashlib.sha256(source.text.encode()).hexdigest()[:8]
        sources.append(
            source.model_copy(
                update={
                    "source_doc_id": f"korea-{article['news_id']}-s600-{index:03d}-{suffix}",
                    "genre": "unclassified",
                }
            )
        )
    article["genre_hint"] = genre_hint(article["title"], article["paragraphs"])
    article["extraction_policy"] = "short_paragraphs_preserved_s600_min200_doccap4"
    article["excerpt_ids"] = [s.source_doc_id for s in sources]
    return article, sources
