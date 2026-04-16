EVALUATION_DATASET = [
    {
        "test_name": "Country A - English - Specific Payment Constraint",
        "request": {
            "question": "Can I pay using my credit card?",
            "country": "A",
            "language": "en",
        },
        "expected_ids": ["a_faq_payment_en"],
        "expected_concept": "Should state that credit cards are not supported, and list cash, UPI, or bank transfer.",
        "expect_empty": False,
    },
    {
        "test_name": "Country A - Hindi - Language Check",
        "request": {
            "question": "मैं अपना खाता कैसे बंद करूं?",
            "country": "A",
            "language": "hi",
        },
        "expected_ids": ["a_faq_account_hi"],
        "expected_concept": "Should explain account closure by contacting support and handling pending orders in Hindi.",
        "expect_empty": False,
    },
    {
        "test_name": "Country B - Spanish - Standard Query",
        "request": {
            "question": "¿Cuál es el tiempo para devolver un producto defectuoso?",
            "country": "B",
            "language": "es",
        },
        "expected_ids": ["b_faq_returns_es", "b_tc_es_v3"],
        "expected_concept": "Should state 30 days for defective items (in Spanish).",
        "expect_empty": False,
    },
    {
        "test_name": "Country B - English - Contradiction Isolation Check",
        "request": {
            "question": "Do you accept cash on delivery?",
            "country": "B",
            "language": "en",
        },
        "expected_ids": ["b_faq_payment_en"],
        "expected_concept": "Must explicitly state cash on delivery is NOT available. (If it says yes, it leaked Country A's data).",
        "expect_empty": False,
    },
    {
        "test_name": "Country C - French - Data Retention Policy",
        "request": {
            "question": "Combien de temps gardez-vous mes données si je ferme mon compte ?",
            "country": "C",
            "language": "fr_CA",
        },
        "expected_ids": ["c_pp_fr_v1", "c_faq_account_fr"],
        "expected_concept": "Should state data is retained for 365 days for tax/audit purposes (in French).",
        "expect_empty": False,
    },
    {
        "test_name": "Country C - English - Multi-tenant Isolation Check",
        "request": {
            "question": "How many days do I have to return an item for a full refund?",
            "country": "C",
            "language": "en",
        },
        "expected_ids": ["c_faq_returns_en"],
        "expected_concept": "Must state 14 days. (If it says 48 hours, it leaked Country A. If it says 7 days, it leaked Country B).",
        "expect_empty": False,
    },
    {
        "test_name": "Language/Country Mismatch - EXPECT EMPTY RESPONSE",
        "request": {
            "question": "write code for sliding window?",
            "country": "C",
            "language": "es",
        },
        "expected_ids": [],
        "expected_concept": "Should return a graceful fallback/empty message since Country C has no Spanish ('es') content.",
        "expect_empty": True,
    },
    {
        "test_name": "Country D - English - Specific Policy Check",
        "request": {
            "question": "How long does it take for my account to be closed after I request it?",
            "country": "D",
            "language": "en",
        },
        "expected_ids": ["d_faq_account_en", "d_tc_en_v1"],
        "expected_concept": "Must state that it takes up to 5 business days. (Unlike Country C, which is immediate).",
        "expect_empty": False,
    },
    {
        "test_name": "Country D - English - Requirement Check",
        "request": {
            "question": "Do I need anything special to return an item?",
            "country": "D",
            "language": "en",
        },
        "expected_ids": ["d_faq_returns_en"],
        "expected_concept": "Must mention that a 'return authorization is required'.",
        "expect_empty": False,
    },
]


import requests

API_URL = "http://127.0.0.1:8000/ask"

passed = 0
for i, test in enumerate(EVALUATION_DATASET):
    print(f"\nRunning Test {i+1}: {test['test_name']}")
    response = requests.post(API_URL, json=test["request"])
    data = response.json()

    #  Check Empty Response Logic
    if test["expect_empty"]:
        if len(data.get("citations", [])) == 0:
            print("✅ PASS: Correctly returned empty/fallback response.")
            passed += 1
        else:
            print(
                f"❌ FAIL: Expected empty response, but got citations: {data['citations']}"
            )
        continue

    # 2. Check Citation Exact Match
    returned_ids = [cite["content_id"] for cite in data.get("citations", [])]
    has_expected_id = any(
        expected_id in returned_ids for expected_id in test["expected_ids"]
    )

    if has_expected_id:
        print(f"✅ PASS: Retrieved expected source document(s).")
        passed += 1
    else:
        print(
            f"❌ FAIL: Expected one of {test['expected_ids']}, but got {returned_ids}"
        )

print(f"\n--- EVALUATION COMPLETE: {passed}/{len(EVALUATION_DATASET)} PASSED ---")
