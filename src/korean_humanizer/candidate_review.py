"""Explainable purpose cues for unused excerpts, never verified genre labels."""

import re

from korean_humanizer.source_scope import exclusion_reason

QUOTES = re.compile(r'"[^"\n]+"|“[^”]+”|「[^」]+」|『[^』]+』')
REPORTING = re.compile(r"고\s*(?:밝혔|말했|전했|설명했|강조했|덧붙였)")
FIRST_PERSON = re.compile(
    r"(?:^|\s)(?:나는|내가|나에게|내게|나를|나의|저는|제가|우리는|우리가)(?![가-힣])"
)
EXPERIENCE = re.compile(
    r"느꼈|겪었|다녀왔|돌아보니|깨달았|어릴\s*때|시절|기억난|기억한다|살았|만났|보았|찾아갔|떠났|도착했|먹었|걸었|배웠"
)
PAST = re.compile(
    r"[가-힣]*(?:았|었|였|했|됐|갔|왔|봤|샀|냈|줬|졌|렸|켰|녔|눴|췄|섰|렀|랐|났|놨|뒀)[가-힣]*"
)
ARGUMENT = re.compile(r"생각한다|생각합니다|해야\s*한다|필요하다|그러므로|따라서|문제는")
DIRECTIONS = re.compile(
    r"신청\s*방법|접수\s*방법|이용\s*방법|제출\s*서류|구비\s*서류|신청\s*자격|참가\s*자격|지원\s*자격|접수\s*기간|신청\s*기간|유의\s*사항|준비물|신청하려|참여하려|희망하는|하세요|하십시오|주시기|바랍니다|신청할\s*수|접수할\s*수"
)
LOGISTICS = re.compile(
    r"까지.{0,12}(?:접수|신청|제출|모집)|(?:접수|신청|모집).{0,16}까지|온라인|인터넷|방문\s*접수|우편|신청서|홈페이지|누리집|문의|연락처|선착순"
)
DIRECT_ACTION = re.compile(
    r"입사지원|신청(?:을|이|은)?\s*(?:하면|하세요|할\s*수|가능)|제출(?:을|이|은)?\s*(?:하면|해야|하세요)|접수(?:를|가|는)?\s*(?:하면|하세요|할\s*수|가능)|이용할\s*수|참가할\s*수|첨부해야|준비해야"
)
NEWS_PROBLEM = re.compile(r"혼란|쇄도|항의|논란|신청도\s*못|지침을\s*받지\s*못")
LEGAL_EVENT = re.compile(
    r"탄원서|구속|가처분|기각|소송|위헌|헌법재판|형사합의|법원|법률\s*해석|세법상"
)
MARKUP = re.compile(
    r"</?[a-zA-Z][^>]*>|(?:font-weight|font-size|text-align)\s*:|[\"'](?:text|color|bold)[\"']\s*:"
)


def assess_candidate(text, *, source_label=None):
    if source_label == "unclassified":
        return {"eligible": False, "reason": "previous_source_label_unclassified"}
    if reason := exclusion_reason(text):
        return {"eligible": False, "reason": reason}
    if MARKUP.search(text):
        return {"eligible": False, "reason": "embedded_markup_needs_source_review"}
    if LEGAL_EVENT.search(text):
        return {"eligible": False, "reason": "legal_dispute_or_professional_procedure_review"}
    if (
        len(re.findall(r"^(?:에디터|포토|의상|슈즈|시계|헤어|메이크업|장소)\s*[:：]", text, re.M))
        >= 3
    ):
        return {"eligible": False, "reason": "credits_listing_needs_source_review"}
    title, separator, body = text.partition("\n")
    heading = separator and len(title) < 100 and not re.search(r"[.!?]|(?:다|요)$", title.strip())
    body = body if heading else text
    unquoted = QUOTES.sub(" ", body)
    features = {
        "first_person": len(FIRST_PERSON.findall(unquoted)),
        "experience": len(EXPERIENCE.findall(unquoted)),
        "past_events": len(PAST.findall(unquoted)),
        "directions": len(DIRECTIONS.findall(unquoted)),
        "logistics": len(LOGISTICS.findall(unquoted)),
        "direct_action": len(DIRECT_ACTION.findall(unquoted)),
        "news_problem": len(NEWS_PROBLEM.findall(unquoted)),
        "reporting": len(REPORTING.findall(unquoted)),
        "argument": len(ARGUMENT.findall(unquoted)),
        "quote_fraction": round(
            sum(len(m.group()) for m in QUOTES.finditer(body)) / max(1, len(body)), 3
        ),
    }
    f = features
    scores = {}
    if (
        f["first_person"]
        and (f["experience"] >= 2 or (f["first_person"] >= 2 and f["past_events"] >= 3))
        and not f["reporting"]
        and f["quote_fraction"] < 0.35
        and f["experience"] + f["past_events"] > 2 * f["argument"]
    ):
        scores["experience_review"] = (
            5 * f["first_person"] + 4 * f["experience"] + min(6, f["past_events"])
        )
    if (
        (
            (f["direct_action"] >= 1 and f["directions"] + f["logistics"] >= 1)
            or (
                f["directions"] >= 2 and (f["logistics"] >= 1 or source_label == "practical_notice")
            )
        )
        and f["reporting"] <= 1
        and f["quote_fraction"] < 0.35
        and f["news_problem"] < 2
    ):
        scores["practical_notice"] = (
            5 * f["directions"] + 3 * f["logistics"] + 4 * f["direct_action"]
        )
    editorial = bool(re.search(r"사설|칼럼|기고|논단|시론|오피니언", title))
    if f["argument"] >= 2 and not f["reporting"] and f["quote_fraction"] < 0.35:
        scores["opinion_column"] = 4 * f["argument"] + 5 * editorial
    if source_label == "explanation_information" and not scores:
        scores["explanation_information"] = 5
    if source_label in scores:
        scores[source_label] += 5
    if not scores:
        return {
            "eligible": False,
            "reason": "insufficient_direct_purpose_cues",
            "features": f,
            "source_label": source_label,
        }
    genre = max(scores, key=scores.get)
    if source_label is not None and source_label != genre:
        return {
            "eligible": False,
            "reason": "purpose_cues_conflict_with_prior_source_label",
            "features": f,
            "source_label": source_label,
            "cue_genre": genre,
        }
    return {
        "eligible": True,
        "genre_hint": genre,
        "score": scores[genre],
        "features": f,
        "source_label": source_label,
        "genre_verified": False,
    }


def assess_np_candidate(text, category, *, source_label=None):
    """Use reconciled NP metadata to recover omitted-subject narratives as hints only."""
    parts = [p.strip() for p in category.split(">")]
    if not parts or parts[0] != "비출판물":
        raise ValueError("reconciled NP category required")
    if any(re.search(r"소설|동화|희곡|극본|상상|^(?:극|시|동시|산문시)$", p) for p in parts[1:]):
        return {"eligible": False, "reason": "np_literary_or_imagined_category_review"}
    decision = assess_candidate(text, source_label=source_label)
    if decision["eligible"] or decision["reason"] != "insufficient_direct_purpose_cues":
        return decision
    if source_label not in {None, "experience_review"}:
        return decision
    narrative = len(parts) > 1 and parts[1] in {"일기", "감상문", "수필"}
    f = decision["features"]
    if (
        narrative
        and f["past_events"] >= 3
        and f["experience"] >= 1
        and f["reporting"] == 0
        and f["quote_fraction"] < 0.35
        and f["past_events"] + f["experience"] > 2 * f["argument"]
    ):
        return {
            "eligible": True,
            "genre_hint": "experience_review",
            "score": 4 * f["experience"] + min(6, f["past_events"]),
            "features": f,
            "source_label": source_label,
            "genre_verified": False,
            "metadata_assisted_hint": True,
            "source_category": category,
        }
    return decision
