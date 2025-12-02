import unittest
from datetime import datetime, timedelta

import main


class TestIPCleanupScheduling(unittest.TestCase):
    def setUp(self):
        self.app = main.app
        self.app.testing = True
        with self.app.app_context():
            main.db.drop_all()
            main.db.create_all()

    def test_cleanup_runs_at_most_once_per_day(self):
        with self.app.test_request_context('/', headers={'X-Forwarded-For': '203.0.113.1'}):
            main._last_ip_cleanup_run = datetime.utcnow() - timedelta(days=2)
            main.log_ip_address()
            first_cleanup_run = main._last_ip_cleanup_run

        with self.app.test_request_context('/', headers={'X-Forwarded-For': '203.0.113.1'}):
            main.log_ip_address()
            self.assertEqual(
                first_cleanup_run,
                main._last_ip_cleanup_run,
                "Cleanup should not rerun within the one-day interval"
            )


if __name__ == '__main__':
    unittest.main()
