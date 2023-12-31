import cflow
import unittest
import subprocess


class TestDocuments(unittest.TestCase):
    def test_generate_documents(self):
        with self.assertRaises(ValueError):
            cflow.generate_documents("docs.txt")
        subprocess.run(["cflow", "docs"])


if __name__ == "__main__":
    unittest.main()
