import unittest
from datetime import datetime, timezone

from app.api.screenings import ScreeningHistoryItem
from app.db.models import Screening


class ScreeningHistoryTests(unittest.TestCase):
    def test_response_schema_serializes_sqlalchemy_screening_rows(self):
        row = Screening(
            id=12,
            user_id=4,
            feature_state_json={},
            risk_probability=0.74,
            risk_label="elevated",
            created_at=datetime(2026, 10, 2, 8, 30, tzinfo=timezone.utc),
            completed=True,
        )
        item = ScreeningHistoryItem.model_validate(row)
        self.assertEqual(item.id, 12)
        self.assertEqual(item.risk_probability, 0.74)
        self.assertEqual(item.risk_label, "elevated")
        self.assertTrue(item.completed)


if __name__ == "__main__":
    unittest.main()
