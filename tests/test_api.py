from fastapi.testclient import TestClient

from api.main import app

c = TestClient(app)
BODY = {"principal_dx": "I5022", "secondary_dx": ["I480", "E1142"], "timing": "early", "institutional": True,
        "oasis": {"M1800": 2, "M1810": 2, "M1820": 2, "M1830": 3, "M1840": 1, "M1850": 2, "M1860": 3,
                  "M1033": ["1", "3", "4", "7"]}, "visits": {"sn": 5, "pt": 4}, "ai_drafted": True}


def test_group_endpoint():
    r = c.post("/group", json=BODY).json()
    assert r["return_code"] == "00" and r["hipps"].startswith("2H") and r["comorbidity"] in {"low", "high"}
    assert r["payment"] > 0 and r["is_lupa"] is False


def test_score_endpoint_explains_itself():
    r = c.post("/score", json=BODY).json()
    assert "risk_score" in r and any("AI draft" in x for x in r["reasons"])


def test_rtp_principal_is_stopped():
    r = c.post("/score", json={**BODY, "principal_dx": "M6281"}).json()
    assert r["action"].startswith("STOP")
