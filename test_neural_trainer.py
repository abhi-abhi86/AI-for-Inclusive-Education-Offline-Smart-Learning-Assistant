"""
Unit tests for Offline Neural Training & Q&A Engine.
Verifies PDF & JSON data loading, vocabulary construction, PyTorch training loop,
.pth model checkpoint compression, and offline question answering.
"""

import os
import json
import tempfile
import unittest
import torch

from data_loader import (
    Vocabulary,
    export_pdf_knowledge_json,
    extract_text_from_pdf,
    load_qa_from_json,
)
from generate_sample_data import create_sample_curriculum_json, create_sample_textbook_pdf
from model import EdgeNeuralQAModel
from qa_engine import OfflineQAModelRunner
from trainer import train_offline_model


class TestNeuralTrainer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pdf_path = "sample_data/test_book.pdf"
        cls.json_path = "sample_data/test_qa.json"
        cls.model_path = "models/test_model.pth"
        create_sample_textbook_pdf(cls.pdf_path)
        create_sample_curriculum_json(cls.json_path)

    @classmethod
    def tearDownClass(cls):
        for path in [cls.pdf_path, cls.json_path, cls.model_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

    def test_pdf_extraction(self):
        chunks = extract_text_from_pdf(self.pdf_path)
        self.assertGreaterEqual(len(chunks), 1)
        self.assertIn("Photosynthesis", chunks[0].text)

    def test_json_loading(self):
        chunks = load_qa_from_json(self.json_path)
        self.assertEqual(len(chunks), 5)
        self.assertIsNotNone(chunks[0].question)
        self.assertIsNotNone(chunks[0].answer)

    def test_pdf_archive_contains_text_and_typed_grounded_questions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = os.path.join(temp_dir, "textbook.pdf")
            json_path = os.path.join(temp_dir, "textbook_knowledge.json")
            create_sample_textbook_pdf(pdf_path)

            result = export_pdf_knowledge_json(pdf_path, json_path)

            self.assertTrue(os.path.exists(json_path))
            self.assertGreater(result["pages"], 0)
            self.assertGreater(result["qa_pairs"], 0)
            with open(json_path, "r", encoding="utf-8") as fp:
                archive = json.load(fp)
            self.assertIn("pages", archive)
            self.assertIn("qa_pairs", archive)
            question_types = {item["question_type"] for item in archive["qa_pairs"]}
            self.assertIn("what", question_types)
            self.assertIn("how", question_types)

            loaded = load_qa_from_json(json_path)
            self.assertGreater(len(loaded), len(archive["qa_pairs"]))
            self.assertTrue(any(chunk.question for chunk in loaded))

    def test_vocabulary_encoding(self):
        vocab = Vocabulary()
        vocab.build_from_texts(["Photosynthesis occurs in chloroplasts.", "Stomata release oxygen."])
        encoded = vocab.encode("Photosynthesis occurs", max_length=10)
        self.assertEqual(len(encoded), 10)
        decoded = vocab.decode(encoded)
        self.assertIn("photosynthesis", decoded)

    def test_model_forward_pass(self):
        vocab = Vocabulary()
        vocab.build_from_texts(["test text for neural network"])
        model = EdgeNeuralQAModel(vocab_size=len(vocab), d_model=32, nhead=2, num_layers=1, dim_feedforward=64)

        q = torch.tensor([[1, 2, 3]])
        ctx = torch.tensor([[1, 2, 3]])
        outputs = model(q, ctx)
        self.assertIn("q_vector", outputs)
        self.assertIn("ctx_vector", outputs)
        self.assertIn("answer_logits", outputs)

    def test_end_to_end_training_and_inference(self):
        # Train for 3 quick epochs
        stats = train_offline_model(
            pdf_paths=[self.pdf_path],
            json_paths=[self.json_path],
            save_path=self.model_path,
            epochs=3,
            batch_size=4,
            d_model=64,
            num_layers=2,
        )
        self.assertTrue(os.path.exists(self.model_path))
        self.assertGreater(stats["num_chunks"], 0)

        # Test offline Q&A runner
        runner = OfflineQAModelRunner(self.model_path)
        resp = runner.answer("What is chlorophyll?")
        self.assertTrue(resp.offline_verified)
        self.assertTrue(resp.generative_verified)
        self.assertGreater(resp.confidence_score, 0.0)
        self.assertIn("chloroplast", resp.answer.lower())

    def test_pure_pdf_training_and_generative_answering(self):
        """Tests that pure PDF textbooks without any JSON questions train and answer properly like generative AI."""
        pdf_only_model = "models/test_pdf_only_model.pth"
        stats = train_offline_model(
            pdf_paths=[self.pdf_path],
            json_paths=[],  # Pure PDF mode: NO pre-written questions
            save_path=pdf_only_model,
            epochs=3,
            batch_size=4,
            d_model=64,
            num_layers=2,
        )
        self.assertTrue(os.path.exists(pdf_only_model))
        self.assertGreater(stats["num_chunks"], 0)

        runner = OfflineQAModelRunner(pdf_only_model)
        # Ask question from PDF text where there was NO prior question
        resp = runner.answer("How do plants produce glucose during photosynthesis?")
        self.assertTrue(resp.offline_verified)
        self.assertTrue(resp.generative_verified)
        self.assertIn("glucose", resp.answer.lower())
        self.assertIn("photosynthesis", resp.answer.lower())

        # Clean up
        if os.path.exists(pdf_only_model):
            os.remove(pdf_only_model)

    def test_greeting_and_conversational_intents(self):
        """Tests that greetings like 'hi', 'who are you', and 'thanks' receive proper conversational responses."""
        runner = OfflineQAModelRunner(self.model_path)
        
        # Test greeting
        resp_hi = runner.answer("hi")
        self.assertIn("welcome", resp_hi.answer.lower())
        self.assertEqual(resp_hi.source_reference, "Conversational AI Engine")

        # Test identity
        resp_who = runner.answer("who are you")
        self.assertIn("assistant", resp_who.answer.lower())

        # Test gratitude
        resp_thanks = runner.answer("thanks a lot")
        self.assertIn("welcome", resp_thanks.answer.lower())

    def test_daily_human_conversations(self):
        """Tests that diverse daily human conversations (jokes, study advice, emotions, philosophy) work properly."""
        conv_model_path = "models/test_conv_model.pth"
        stats = train_offline_model(
            pdf_paths=[self.pdf_path],
            json_paths=["sample_data/daily_conversations.json", self.json_path],
            save_path=conv_model_path,
            epochs=2,
            batch_size=8,
            d_model=64,
            num_layers=2,
        )
        self.assertTrue(os.path.exists(conv_model_path))
        runner = OfflineQAModelRunner(conv_model_path)

        # 1. Joke
        resp_joke = runner.answer("Tell me a joke!")
        self.assertIn("atom", resp_joke.answer.lower())
        self.assertTrue(resp_joke.offline_verified)

        # 2. Study advice
        resp_study = runner.answer("How can I study effectively and remember more?")
        self.assertIn("recall", resp_study.answer.lower())

        # 3. Emotions & Empathy
        resp_sad = runner.answer("I am feeling sad today.")
        self.assertTrue("sorry" in resp_sad.answer.lower() or "gentle" in resp_sad.answer.lower())

        # 4. Curiosity
        resp_sky = runner.answer("Why is the sky blue?")
        self.assertIn("scattering", resp_sky.answer.lower())

        # 5. Check size is well under 80 MB
        size_mb = os.path.getsize(conv_model_path) / (1024.0 * 1024.0)
        self.assertLess(size_mb, 80.0)

        # Clean up
        if os.path.exists(conv_model_path):
            os.remove(conv_model_path)


if __name__ == "__main__":
    unittest.main()
