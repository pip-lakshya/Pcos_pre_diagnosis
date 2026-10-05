import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.agent.research_provider import KnowledgeBaseProvider, PERSONALIZED_INSIGHTS_DISCLAIMER
from app.api.research import _saved_details_answer, build_screening_context


class ResearchContextTests(unittest.TestCase):
    def test_personal_details_requests_return_saved_answers_without_zero_placeholders(self):
        screening = SimpleNamespace(
            risk_label="lower", risk_probability=0.29,
            feature_state_json={"features": {
                "age": 20, "weight_kg": 55.0, "height_cm": 165.0,
                "cycle_irregular": 1, "cycle_length_days": 28,
                "acne": 1, "hip_inch": 0.0, "waist_inch": 0.0,
            }},
        )
        answer = _saved_details_answer("tell me the details you filled", screening)
        self.assertIn("Age: 20", answer)
        self.assertIn("Reported weight: 55 kg", answer)
        self.assertIn("29% (lower range)", answer)
        self.assertNotIn("Hip measurement", answer)
        self.assertIsNone(_saved_details_answer("what foods are good for PCOS?", screening))

    def test_context_uses_saved_label_and_only_top_reported_feature_names(self):
        screening = SimpleNamespace(
            risk_label="elevated",
            risk_probability=0.74,
            feature_state_json={
                "features": {"cycle_irregular": 1, "skin_darkening": 1, "acne": 0},
                "feature_importances": {"cycle_irregular": 0.4, "skin_darkening": 0.3, "acne": 0.2},
            },
        )
        context = build_screening_context(screening)
        self.assertEqual(context["screening_level"], "elevated")
        self.assertEqual(context["reported_positive_features_among_top_inputs"], ["irregular cycles", "reported skin darkening"])
        self.assertNotIn("risk_probability", context)
        self.assertNotIn("features", context)

    def test_every_research_answer_and_system_prompt_include_persistent_disclaimer(self):
        calls = []
        def fake_create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=(
                "Your saved result was elevated, and you reported irregular cycles and skin darkening. "
                "Those inputs are not proof of a cause. [symptoms.md]"
            )))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
        with patch("app.agent.research_provider.OpenAI", return_value=client):
            provider = KnowledgeBaseProvider()
            context = {
                "screening_level": "elevated",
                "top_contributing_features": ["irregular cycles", "reported skin darkening"],
                "reported_positive_features_among_top_inputs": ["irregular cycles", "reported skin darkening"],
            }
            first = provider.answer("Could irregular cycles and skin darkening relate to PCOS?", context)
            second = provider.answer("What can cause PCOS?", context)
        for answer in (first, second):
            self.assertIn("Your saved result was elevated", answer)
            self.assertIn(PERSONALIZED_INSIGHTS_DISCLAIMER, answer)
        system_prompt = calls[0]["messages"][0]["content"]
        user_prompt = calls[0]["messages"][1]["content"]
        self.assertIn("not affiliated with or partnered", system_prompt)
        self.assertIn("does not provide medical advice", system_prompt)
        self.assertIn("licensed professional", system_prompt)
        self.assertIn("screening_level': 'elevated'", user_prompt)
        self.assertIn("irregular cycles", user_prompt)
        self.assertIn("reported skin darkening", user_prompt)
        self.assertNotIn("0.74", user_prompt)


if __name__ == "__main__":
    unittest.main()
