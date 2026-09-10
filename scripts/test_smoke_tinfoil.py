"""Offline checks for the provider smoke contract: python3 -m unittest discover -s scripts."""
import contextlib
import io
import os
from pathlib import Path
import subprocess
import types
import unittest
from unittest.mock import mock_open, patch

SCRIPT = Path(__file__).with_name("smoke_tinfoil.sh")
CHECKS = SCRIPT.read_text().split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]


class SmokeModeTests(unittest.TestCase):
    def run_checks(self, mode, failing_model=None):
        calls = []

        def post(url, *, headers, json, timeout):
            model = json["model"]
            calls.append(model)
            status = 404 if model in (failing_model, "definitely-not-a-real-model") else 200
            body = {"model": model, "choices": [{"finish_reason": "stop", "message": {
                "content": "red square" if model == "vision" else "OK"}}],
                "data": [{"embedding": [0.0] * 768}]}
            return types.SimpleNamespace(status_code=status, text="missing", json=lambda: body)

        env = dict(TINFOIL_API_URL="http://unused/v1", TINFOIL_API_KEY="test",
                   TINFOIL_MODEL="chat", TINFOIL_EMBEDDING_MODEL="embedding",
                   TINFOIL_VISION_MODEL="vision", TINFOIL_REASONING_EFFORT="low",
                   SAGE_SMOKE_MODE=mode)
        output = io.StringIO()
        with patch.dict(os.environ, env), patch.dict("sys.modules", {"requests": types.SimpleNamespace(post=post)}), \
                patch("subprocess.run"), patch("builtins.open", mock_open(read_data=b"image")), \
                contextlib.redirect_stdout(output):
            exec(compile(CHECKS, str(SCRIPT), "exec"), {})
        return calls, output.getvalue()

    def test_enclave_excludes_vision_but_runs_shared_checks(self):
        calls, output = self.run_checks("enclave", failing_model="vision")
        self.assertEqual(calls, ["chat", "embedding", "definitely-not-a-real-model"])
        self.assertIn("NOT TESTED vision", output)
        self.assertNotIn("PASS vision", output)

    def test_full_checks_vision_and_fails_when_unavailable(self):
        calls, output = self.run_checks("full")
        self.assertIn("vision", calls)
        self.assertIn("PASS vision", output)
        with self.assertRaisesRegex(SystemExit, "FAIL vision status 404"):
            self.run_checks("full", failing_model="vision")

    def test_enclave_does_not_mask_embedding_failure(self):
        with self.assertRaisesRegex(SystemExit, "FAIL embeddings status 404"):
            self.run_checks("enclave", failing_model="embedding")

    def test_arguments_are_validated_before_starting_containers(self):
        for args, status in [(["--help"], 0), (["--mode", "typo"], 1), (["enclave"], 1)]:
            with self.subTest(args=args):
                result = subprocess.run(["bash", str(SCRIPT), *args], capture_output=True,
                                        env={**os.environ, "CONTAINER_ENGINE": "does-not-exist"})
                self.assertEqual(result.returncode, status)
                self.assertIn(b"Usage:", result.stdout + result.stderr)
