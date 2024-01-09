import unittest
import subprocess

from pathlib import Path


file_folder = Path(__file__).parent
examples_folder = file_folder.parent / "examples"


class TestExample(unittest.TestCase):
    def test_code_snippets(self):
        for path in examples_folder.glob("*.py"):
            # skip sd examples
            if path.stem.startswith("sd"):
                continue
            # skip openai examples
            if path.stem.startswith("openai"):
                continue
            subprocess.run(["python", str(path)], check=True)

    def test_workflows(self):
        for path in (examples_folder / "workflows").rglob("*.json"):
            # skip running openai examples
            if path.parent.stem != "openai":
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
            # but all examples can be rendered
            subprocess.run(
                ["cflow", "render", "-f", str(path), "-o", f"{path.stem}.png"],
                check=True,
            )


if __name__ == "__main__":
    unittest.main()
