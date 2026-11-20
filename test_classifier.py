import json
import pandas as pd
from classifier import classify_dataframe, parse_response


def fake_llm(prompt, key):
    ids = [int(l.split(".")[0]) for l in prompt.split("Feedback items:")[1].strip().splitlines() if l[0].isdigit()]
    return "```json\n" + json.dumps([{"id": i, "category": "delivery", "priority": "high",
                                       "sentiment": "negative", "reason": "x"} for i in ids]) + "\n```"


def test_parse_fenced():
    assert parse_response('```json\n[{"id":1}]\n```') == [{"id": 1}]


def test_classify_batches_and_normalises():
    df = pd.read_csv("sample_feedback.csv")
    out = classify_dataframe(df, "feedback", "k", batch_size=7, llm=fake_llm)
    assert len(out) == 20
    assert set(out["category"]) == {"Other"} or "Delivery" in set(out["category"])
    assert set(out["priority"]) == {"High"}


def test_bad_json_falls_back():
    df = pd.DataFrame({"feedback": ["a", "b"]})
    out = classify_dataframe(df, "feedback", "k", llm=lambda p, k: "not json")
    assert list(out["reason"]) == ["parse error"] * 2
