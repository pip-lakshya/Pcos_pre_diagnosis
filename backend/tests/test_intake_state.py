"""Regression coverage for session slots and confirmation classification."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.agent.session_store import Session, merge_features, missing
from app.agent import feature_extractor
from app.api.chat import _is_affirmative_confirmation
from app.agent.llm_client import _references_missing, fallback_question, next_question


class IntakeStateTests(unittest.TestCase):
    def test_yes_no_tool_values_are_normalized_before_session_validation(self):
        self.assertIs(feature_extractor._normalize_boolean("yes"), True)
        self.assertIs(feature_extractor._normalize_boolean("No."), False)
        self.assertIsNone(feature_extractor._normalize_boolean("maybe"))

        function = SimpleNamespace(
            name="extract_pcos_features",
            arguments='{"weight_gain":"no", "acne":"yes", "hair_loss":"maybe"}',
        )
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            tool_calls=[SimpleNamespace(function=function)]
        ))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: response
        )))
        with patch.object(feature_extractor, "OpenAI", return_value=client), patch.object(
            feature_extractor.settings, "nvidia_api_key", "test-key"
        ):
            updates = feature_extractor.extract("No weight gain, yes acne", {}, [])

        self.assertEqual(updates, {"weight_gain": False, "acne": True})
        session = Session(user_id=7)
        merge_features(session, updates)
        self.assertEqual(session.values["weight_gain"], 0)
        self.assertEqual(session.values["acne"], 1)
        self.assertNotIn("weight_gain", missing(session))
        self.assertNotIn("acne", missing(session))

    def test_missing_tool_call_recovers_short_answer_from_last_question(self):
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[]))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: response
        )))
        history = [{"role": "assistant", "content": "Do you exercise regularly?"}]
        with patch.object(feature_extractor, "OpenAI", return_value=client), patch.object(
            feature_extractor.settings, "nvidia_api_key", "test-key"
        ):
            updates = feature_extractor.extract("no", {}, history)
        self.assertEqual(updates, {"regular_exercise": False})

        history = [{"role": "assistant", "content": "What is your weight in kilograms?"}]
        with patch.object(feature_extractor, "OpenAI", return_value=client), patch.object(
            feature_extractor.settings, "nvidia_api_key", "test-key"
        ):
            updates = feature_extractor.extract("55", {}, history)
        self.assertEqual(updates, {"weight_kg": 55.0})

    def test_short_answer_bypasses_remote_extraction_call(self):
        history = [{"role": "assistant", "content": "Do you currently have acne?"}]
        with patch.object(feature_extractor, "OpenAI", side_effect=AssertionError("remote call")):
            updates = feature_extractor.extract("yes", {}, history)
        self.assertEqual(updates, {"acne": True})

    def test_followup_question_is_local_and_does_not_repeat_disclaimer(self):
        question = fallback_question(["age", "weight_kg"])
        self.assertEqual(question, "How old are you?")
        self.assertNotIn("screening estimate", question.lower())

    def test_empty_explanation_response_uses_safe_local_result_text(self):
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: response
        )))
        with patch("app.agent.llm_client.OpenAI", return_value=client), patch.object(
            feature_extractor.settings, "nvidia_api_key", "test-key"
        ):
            from app.agent.llm_client import explain_prediction
            explanation = explain_prediction(0.73, "elevated", {"acne": 0.2}, 0.82, [])

        self.assertIn("73%", explanation)
        self.assertIn("not a diagnosis", explanation.lower())
        self.assertIn("doctor", explanation.lower())
        self.assertIn("strongly agree", explanation.lower())

    def test_explicit_false_and_zero_are_not_missing(self):
        session = Session(user_id=7)
        merge_features(session, {
            "acne": False,
            "cycle_irregular": False,
            "regular_exercise": False,
            "age": 0,
            "cycle_length_days": 0,
        })
        absent = missing(session)
        for name in ("acne", "cycle_irregular", "regular_exercise", "age", "cycle_length_days"):
            self.assertNotIn(name, absent)
        self.assertEqual(session.values["acne"], 0)
        self.assertEqual(session.values["age"], 0)

    def test_fallback_does_not_repeat_a_field_answered_false(self):
        session = Session(user_id=7)
        merge_features(session, {"age": 24, "acne": False})
        question = fallback_question(missing(session))
        self.assertNotIn("acne", missing(session))
        self.assertNotIn("acne", question.lower())

    def test_correction_to_false_is_detected_as_a_change(self):
        session = Session(user_id=7, values={"acne": 1})
        changed = merge_features(session, {"acne": False})
        self.assertEqual(changed, {"acne"})
        self.assertEqual(session.values["acne"], 0)

    def test_confirmation_accepts_natural_affirmative_and_rejects_correction(self):
        self.assertTrue(_is_affirmative_confirmation("Yes, that summary sounds right to me."))
        self.assertTrue(_is_affirmative_confirmation("Okay, that's accurate."))
        self.assertFalse(_is_affirmative_confirmation("No, actually my acne is not persistent."))

    def test_followup_must_reference_an_outstanding_field(self):
        self.assertTrue(_references_missing("How many days are there between periods?", ["cycle_length_days"]))
        self.assertFalse(_references_missing("Have you noticed acne?", ["age"]))

    def test_hip_and_waist_are_optional_and_not_requested(self):
        session = Session(user_id=7)
        self.assertNotIn("hip_inch", missing(session))
        self.assertNotIn("waist_inch", missing(session))
        self.assertEqual(missing(session)[0], "age")

    def test_hindi_yes_no_and_skip_are_recognized(self):
        self.assertIs(feature_extractor._normalize_boolean("haan"), True)
        self.assertIs(feature_extractor._normalize_boolean("नहीं"), False)
        self.assertTrue(feature_extractor.is_skip_answer("mujhe waist ka pata nahi"))
        self.assertTrue(feature_extractor.is_skip_answer("मुझे नहीं पता"))
        self.assertEqual(
            feature_extractor.skip_targets(
                "waist ka nahi pata", [{"role": "assistant", "content": "Do you exercise regularly?"}]
            ),
            {"waist_inch"},
        )

    def test_short_measurements_convert_explicit_units(self):
        history = [{"role": "assistant", "content": "Do you know your hip measurement in inches?"}]
        self.assertEqual(feature_extractor.extract("90 cm", {}, history), {"hip_inch": 90 / 2.54})
        history = [{"role": "assistant", "content": "What is your height in centimeters?"}]
        self.assertEqual(feature_extractor.extract("60 inches", {}, history), {"height_cm": 152.4})
        converted = feature_extractor._normalize_explicit_units(
            "My waist is 80 cm and my hip is 100 cm", {"waist_inch": 80, "hip_inch": 100}
        )
        self.assertAlmostEqual(converted["waist_inch"], 80 / 2.54)
        self.assertAlmostEqual(converted["hip_inch"], 100 / 2.54)
        self.assertEqual(feature_extractor._normalize_explicit_units("my height is 165 cms", {})["height_cm"], 165)

    def test_followup_question_adapts_to_hinglish(self):
        history = [{"role": "user", "content": "meri age 24 hai"}]
        self.assertEqual(next_question(history, ["weight_kg"]), "Aapka wazan kilogram mein kitna hai?")

    def test_hindi_confirmation_is_affirmative(self):
        self.assertTrue(_is_affirmative_confirmation("haan, sahi hai"))
        self.assertFalse(_is_affirmative_confirmation("nope, acne is wrong"))


if __name__ == "__main__":
    unittest.main()
