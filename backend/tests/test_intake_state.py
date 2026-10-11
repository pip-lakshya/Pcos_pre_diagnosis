"""Regression coverage for session slots and confirmation classification."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.agent.session_store import Session, merge_features, missing
from app.agent import feature_extractor
from app.api.chat import (
    _explicit_correction, _is_affirmative_confirmation, _is_correction_message,
    _is_restart_intent, _is_summary_request, _lookup_reply, _summary_text, chat,
)
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
        self.assertNotIn("agree", explanation.lower())

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

    def test_labeled_weight_corrections_are_recognized_without_llm(self):
        examples = (
            ("mera sahi wajan 56 hai", 56.0),
            ("my correct weight is 56", 56.0),
            ("update my weight to 56", 56.0),
            ("change my weight from 55 to 56 kg", 56.0),
        )
        for text, expected in examples:
            with self.subTest(text=text):
                updates = feature_extractor.recover_labeled_numeric_update(text)
                self.assertEqual(updates, {"weight_kg": expected})
                self.assertTrue(_is_correction_message(text))
                with patch.object(feature_extractor, "OpenAI", side_effect=AssertionError("remote call")):
                    self.assertEqual(feature_extractor.extract(text, {"weight_kg": 55.0}, []), updates)

    def test_labeled_value_can_fill_a_previously_missing_field(self):
        session = Session(user_id=9)
        update = feature_extractor.recover_labeled_numeric_update("I forgot to mention my age is 24")
        self.assertEqual(update, {"age": 24})
        merge_features(session, update)
        self.assertEqual(session.values["age"], 24)
        self.assertNotIn("age", missing(session))

    def test_new_chat_phrases_restart_a_screening(self):
        for phrase in ("restart", "start over", "new chat", "test again", "phir se shuru karo", "नया टेस्ट"):
            with self.subTest(phrase=phrase):
                self.assertTrue(_is_restart_intent(phrase))

    def test_field_correction_is_not_treated_as_confirmation(self):
        self.assertTrue(_is_correction_message("my correct weight is 56"))
        self.assertFalse(_is_affirmative_confirmation("my correct weight is 56"))
        self.assertTrue(_is_correction_message("my acne is no"))

    def test_followup_must_reference_an_outstanding_field(self):
        self.assertTrue(_references_missing("How many days are there between periods?", ["cycle_length_days"]))
        self.assertFalse(_references_missing("Have you noticed acne?", ["age"]))

    def test_hip_and_waist_are_optional_followup_questions(self):
        session = Session(user_id=7)
        self.assertIn("hip_inch", missing(session))
        self.assertIn("waist_inch", missing(session))
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

    def test_summary_intent_and_partial_summary_are_state_based(self):
        self.assertTrue(_is_summary_request("Can you show my updated summary?"))
        self.assertTrue(_is_summary_request("Meri summary dikhao"))
        session = Session(user_id=81, values={"age": 25, "weight_kg": 60.0})
        summary = _summary_text(session.values, session.skipped, missing(session), None)
        self.assertIn("Age: 25 years", summary)
        self.assertIn("Weight: 60 kg", summary)
        self.assertIn("Height: Not provided", summary)
        self.assertIn("Not calculated yet", summary)

    def test_field_lookup_returns_exact_value_or_missing(self):
        session = Session(user_id=81, values={"hip_inch": 38.0, "acne": 0})
        self.assertIn("38 inches", _lookup_reply("What was my hip measurement?", session, None))
        self.assertIn("no", _lookup_reply("Did I report acne?", session, None))
        self.assertIn("recorded", _lookup_reply("What was my waist measurement?", session, None))

    def test_ambiguous_and_sometimes_corrections_ask_for_clarification(self):
        self.assertEqual(_explicit_correction("Change that answer"), ({}, "Which answer would you like to correct, and what is the correct value?", None))
        updates, clarification, _ = _explicit_correction("Update my acne answer to sometimes")
        self.assertEqual(updates, {})
        self.assertIn("yes or no", clarification)
        self.assertEqual(_explicit_correction("yes"), ({}, None, None))
        self.assertEqual(_explicit_correction("yes, this is correct"), ({}, None, None))

    def test_successful_correction_updates_state_and_recomputes_summary_and_bmi(self):
        from app.models.schemas import ChatRequest, PredictResponse

        session = Session(user_id=918, values={"age": 25, "weight_kg": 60.0, "height_cm": 160.0,
            "cycle_irregular": 1, "cycle_length_days": 20, "weight_gain": 0,
            "hair_growth": 0, "skin_darkening": 0, "hair_loss": 0, "acne": 0,
            "fast_food": 0, "regular_exercise": 1})
        # Isolate the test session from the process-local session registry.
        from app.agent.session_store import SESSIONS
        SESSIONS[(918, "correction-test")] = session
        response = chat(ChatRequest(session_id="correction-test", message="my correct weight is 56 kg"),
            SimpleNamespace(id=918), SimpleNamespace())
        self.assertEqual(session.values["weight_kg"], 56.0)
        self.assertEqual(session.values["bmi"], 21.87)
        self.assertEqual(response.message_type, "summary")
        self.assertIn("Updated weight kg", response.reply)
        self.assertIn("Weight: 56 kg", response.reply)
        self.assertEqual(session.values["age"], 25)
        SESSIONS.pop((918, "correction-test"), None)

    def test_summary_request_does_not_start_or_mutate_intake(self):
        from app.models.schemas import ChatRequest
        from app.agent.session_store import SESSIONS

        SESSIONS[(919, "summary-test")] = Session(user_id=919)
        response = chat(ChatRequest(session_id="summary-test", message="Show my summary"),
            SimpleNamespace(id=919), SimpleNamespace())
        self.assertFalse(response.complete)
        self.assertFalse(response.collected)
        self.assertEqual(response.message_type, "summary")
        self.assertFalse(SESSIONS[(919, "summary-test")].screening_started)
        SESSIONS.pop((919, "summary-test"), None)

    def test_completed_correction_saves_new_snapshot_and_keeps_old_screening_immutable(self):
        from app.db.models import Screening
        from app.models.schemas import ChatRequest, PredictResponse
        from app.agent.session_store import SESSIONS

        original_features = {"age": 25, "weight_kg": 60.0, "height_cm": 160.0,
            "cycle_irregular": 1, "cycle_length_days": 20, "weight_gain": 0,
            "hair_growth": 0, "skin_darkening": 0, "hair_loss": 0, "acne": 0,
            "fast_food": 0, "regular_exercise": 1, "hip_inch": 36.0, "waist_inch": 30.0,
            "bmi": 23.44, "waist_hip_ratio": 0.833}
        old = Screening(id=40, user_id=920, feature_state_json={"features": dict(original_features)},
            risk_probability=0.4, risk_label="lower", completed=True)
        class Query:
            def filter(self, *args): return self
            def first(self): return old
        class DB:
            rows=[]
            def query(self, *args): return Query()
            def add(self, row): self.rows.append(row)
            def commit(self): self.rows[-1].id=41
            def refresh(self, row): pass
            def rollback(self): pass
        db=DB()
        session=Session(user_id=920, values=dict(original_features), screening_id=40, screening_started=True)
        SESSIONS[(920, "completed-correction")] = session
        output=PredictResponse(probability=0.62, risk_label="elevated", feature_importances={},
            agreement_score=0.5, model_probabilities={})
        with patch("app.api.chat.predict", return_value=output) as predict_mock, patch(
            "app.ml.predictor.load_assets", return_value=(None, list(original_features))
        ):
            response=chat(ChatRequest(session_id="completed-correction", message="my hip is actually 38 inches"),
                SimpleNamespace(id=920), db)
        self.assertEqual(predict_mock.call_count, 1)
        self.assertEqual(response.screening_id, 41)
        self.assertEqual(response.prediction.probability, 0.62)
        self.assertEqual(old.feature_state_json["features"]["hip_inch"], 36.0)
        self.assertEqual(db.rows[0].feature_state_json["features"]["hip_inch"], 38.0)
        self.assertEqual(session.values["hip_inch"], 38.0)
        SESSIONS.pop((920, "completed-correction"), None)

    def test_completed_correction_failure_does_not_mutate_session(self):
        from app.db.models import Screening
        from app.models.schemas import ChatRequest
        from app.agent.session_store import SESSIONS
        from fastapi import HTTPException

        features={"age":25,"weight_kg":60.0,"height_cm":160.0,"hip_inch":36.0}
        old=Screening(id=42,user_id=921,feature_state_json={"features":features},risk_probability=.4,risk_label="lower",completed=True)
        class Query:
            def filter(self,*args): return self
            def first(self): return old
        class DB:
            def query(self,*args): return Query()
            def rollback(self): pass
        session=Session(user_id=921,values=dict(features),screening_id=42)
        SESSIONS[(921,"failed-correction")]=session
        with patch("app.api.chat.predict", side_effect=RuntimeError("inference unavailable")), patch(
            "app.ml.predictor.load_assets", return_value=(None, list(features))
        ), self.assertRaises(HTTPException):
            chat(ChatRequest(session_id="failed-correction",message="my hip is actually 38 inches"),
                SimpleNamespace(id=921),DB())
        self.assertEqual(session.values["hip_inch"],36.0)
        self.assertEqual(session.screening_id,42)
        SESSIONS.pop((921,"failed-correction"),None)


if __name__ == "__main__":
    unittest.main()
