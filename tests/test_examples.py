import unittest
import subprocess

from pathlib import Path


file_folder = Path(__file__).parent
examples_folder = file_folder.parent / "examples"


class TestExample(unittest.TestCase):
    def test_code_snippets(self):
        for path in examples_folder.glob("*.py"):
            subprocess.run(["python", str(path)], check=True)

    def test_workflows(self):
        for path in (examples_folder / "workflows").rglob("*.json"):
            subprocess.run(
                [
                    "cflow",
                    "execute",
                    "-f",
                    str(path),
                    "-o",
                    f"{path.stem}_results.json",
                ],
                check=True,
            )
            subprocess.run(
                ["cflow", "render", "-f", str(path), "-o", f"{path.stem}.png"],
                check=True,
            )


if __name__ == "__main__":
    unittest.main()
