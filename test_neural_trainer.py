"""
Unit tests for Offline Neural Training & Q&A Engine.
Verifies PDF & JSON data loading, vocabulary construction, PyTorch training loop,
.pth model checkpoint compression, and offline question answering.
"""

import os
import unittest
import torch

from data_loader import Vocabulary, extract_text_from_pdf, load_qa_from_json
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

    def test_pdf_extraction(self):
        chunks = extract_text_from_pdf(self.pdf_path)
        self.assertGreaterEqual(len(chunks), 1)
        self.assertIn("Photosynthesis", chunks[0].text)

    def test_json_loading(self):
        chunks = load_qa_from_json(self.json_path)
        self.assertEqual(len(chunks), 5)
        self.assertIsNotNone(chunks[0].question)
        self.assertIsNotNone(chunks[0].answer)

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
        self.assertGreater(resp.confidence_score, 0.0)
        self.assertIn("chloroplast", resp.answer.lower())


if __name__ == "__main__":
    unittest.main()

