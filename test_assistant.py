"""
Unit tests for Offline Smart Learning Assistant.
Tests passage validation, prompt generation, offline deterministic engine,
and edge telemetry reporting.
"""

import unittest
from engine import OfflineDeterministicEngine, get_inference_engine
from metrics import PerformanceTimer, estimate_token_count, get_peak_memory_mb
from prompt_templates import build_prompt, parse_learning_aid, validate_passage


class TestOfflineAssistant(unittest.TestCase):

    def setUp(self):
        self.sample_passage = (
            "Photosynthesis is the fundamental biological process by which green plants, algae, "
            "and certain bacteria transform light energy into chemical energy. During this process, "
            "plants absorb sunlight using a green pigment called chlorophyll, which is located inside "
            "cell structures termed chloroplasts. Plants take in carbon dioxide from the surrounding "
            "air through tiny microscopic pores in their leaves known as stomata, while simultaneously "
            "absorbing water and essential mineral nutrients from the soil through their root systems. "
            "Powered by the captured solar energy, a chemical reaction occurs inside the chloroplasts "
            "that converts carbon dioxide and water into glucose. As an essential byproduct, oxygen gas "
            "is released back into the atmosphere through the stomata. This liberated oxygen is vital "
            "for the survival of aerobic organisms on Earth, including humans."
        )

    def test_passage_validation(self):
        is_valid, count, msg = validate_passage(self.sample_passage)
        self.assertTrue(is_valid)
        self.assertGreaterEqual(count, 100)
        self.assertLessEqual(count, 350)

    def test_empty_passage_validation(self):
        is_valid, count, msg = validate_passage("")
        self.assertFalse(is_valid)
        self.assertEqual(count, 0)

    def test_prompt_construction(self):
        prompt = build_prompt(self.sample_passage)
        self.assertIn("Photosynthesis", prompt)
        self.assertIn("### SUMMARY", prompt)
        self.assertIn("### QUIZ QUESTIONS", prompt)

    def test_deterministic_engine_generation(self):
        engine = OfflineDeterministicEngine()
        output, telemetry = engine.generate(self.sample_passage)

        self.assertIn("### SUMMARY", output)
        self.assertIn("### QUIZ QUESTIONS", output)
        self.assertIn("Photosynthesis", output)
        self.assertIn("chloroplasts", output.lower())

        # Verify telemetry properties
        self.assertTrue(telemetry.offline_verified)
        self.assertGreater(telemetry.total_tokens, 0)
        self.assertGreater(telemetry.tokens_per_second, 0.0)
        self.assertGreater(telemetry.peak_ram_mb, 0.0)

    def test_parse_learning_aid(self):
        engine = OfflineDeterministicEngine()
        output, _ = engine.generate(self.sample_passage)
        parsed = parse_learning_aid(output)
        self.assertTrue(len(parsed["summary"]) > 0)
        self.assertTrue(len(parsed["quiz"]) > 0)

    def test_memory_and_timing_metrics(self):
        timer = PerformanceTimer()
        timer.start()
        timer.mark_first_token()
        duration = timer.stop()
        self.assertGreaterEqual(duration, 0.0)
        self.assertIsNotNone(timer.ttft)

        ram = get_peak_memory_mb()
        self.assertGreater(ram, 0.0)


if __name__ == "__main__":
    unittest.main()

