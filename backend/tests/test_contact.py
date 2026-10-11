import unittest
from unittest.mock import AsyncMock, patch

import httpx

from app.config import settings
from app.main import app
from app.services.email import EmailProviderError


class ContactWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver")
        self.settings = patch.multiple(
            settings,
            email_provider="smtp",
            admin_notify_email="team@example.com",
            smtp_user="sender@example.com",
            smtp_app_password="test-only",
        )
        self.settings.start()

    async def asyncTearDown(self):
        self.settings.stop()
        await self.client.aclose()

    async def test_preflight_allows_both_production_origins(self):
        for origin in ("https://narisaarthi.site", "https://www.narisaarthi.site"):
            with self.subTest(origin=origin):
                response = await self.client.options(
                    "/contact",
                    headers={
                        "Origin": origin,
                        "Access-Control-Request-Method": "POST",
                        "Access-Control-Request-Headers": "content-type",
                    },
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers.get("access-control-allow-origin"), origin)
                self.assertIn("POST", response.headers.get("access-control-allow-methods", ""))
                self.assertNotIn("access-control-allow-credentials", response.headers)

    async def test_preflight_rejects_unlisted_origin(self):
        response = await self.client.options(
            "/contact",
            headers={
                "Origin": "https://attacker.example",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        self.assertNotEqual(response.headers.get("access-control-allow-origin"), "https://attacker.example")

    async def test_valid_submission_returns_success_only_after_provider_accepts(self):
        with patch("app.api.contact.run_in_threadpool", new=AsyncMock(return_value=True)) as deliver:
            response = await self.client.post(
                "/contact",
                json={"name": "A User", "email": "a@example.com", "subject": "Hello", "message": "Please contact our team."},
                headers={"Origin": "https://narisaarthi.site"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertIn("sent", response.json()["message"])
        deliver.assert_awaited_once()
        self.assertEqual(response.headers.get("access-control-allow-origin"), "https://narisaarthi.site")

    async def test_invalid_payload_is_rejected(self):
        response = await self.client.post("/contact", json={"name": "", "email": "bad", "subject": "", "message": ""})
        self.assertEqual(response.status_code, 422)

    async def test_missing_provider_configuration_returns_503(self):
        with patch.object(settings, "admin_notify_email", ""), patch("app.api.contact.send_contact_email") as send_email:
            response = await self.client.post(
                "/contact",
                json={"name": "A User", "email": "a@example.com", "subject": "Hello", "message": "Please contact our team."},
            )
        self.assertEqual(response.status_code, 503)
        send_email.assert_not_called()

    async def test_provider_failure_returns_safe_error_and_not_success(self):
        with patch("app.api.contact.run_in_threadpool", new=AsyncMock(return_value=False)):
            response = await self.client.post(
                "/contact",
                json={"name": "A User", "email": "a@example.com", "subject": "Hello", "message": "Please contact our team."},
                headers={"Origin": "https://www.narisaarthi.site"},
            )
        self.assertEqual(response.status_code, 502)
        self.assertNotIn("provider", response.text.lower())
        self.assertEqual(response.headers.get("access-control-allow-origin"), "https://www.narisaarthi.site")

    async def test_delivery_worker_converts_provider_error_to_failure(self):
        from app.api.contact import _deliver_contact_email

        with patch("app.api.contact.send_contact_email", side_effect=EmailProviderError("private provider detail")):
            self.assertFalse(_deliver_contact_email("A User", "a@example.com", "Hello", "Please contact our team."))

    async def test_contact_email_uses_admin_recipient_and_visitor_reply_to(self):
        from app.services import email

        with patch.object(email, "_send_one") as send_one:
            email.send_contact_email("A User", "a@example.com", "Hello", "Please contact our team.")
        args, kwargs = send_one.call_args
        self.assertEqual(args[0], "team@example.com")
        self.assertIn("A User", args[2])
        self.assertIn("a@example.com", args[2])
        self.assertIn("Please contact our team.", args[2])
        self.assertEqual(kwargs["reply_to"], "a@example.com")


if __name__ == "__main__":
    unittest.main()
