import io
import platform

from setuptools import find_packages, setup

with io.open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

install_requires = [
    "Flask==3.0.3",
    "numpy==1.26.4",
    "mss==9.0.1",
    "Pillow==10.3.0",
    # ChromaDB — embedded vector DB with built-in semantic search (no Docker needed)
    "chromadb>=0.5.0",
    # Sentence-transformers — used internally by ChromaDB for local embeddings
    "sentence-transformers>=3.0.0",
    # OpenAI-compatible client — used to call OpenRouter vision models
    "openai>=1.30.0",
]

extras_require = {
    "windows": ["pywin32", "psutil"],
    "macos": ["pyobjc==10.3"],
    "linux": [],
}

current_os = platform.system().lower()
if current_os.startswith("win"):
    current_os = "windows"
elif current_os == "darwin":
    current_os = "macos"
elif current_os == "linux":
    current_os = "linux"
else:
    current_os = None

if current_os and current_os in extras_require:
    install_requires.extend(extras_require[current_os])

setup(
    name="OpenRecall",
    version="0.9",
    packages=find_packages(),
    install_requires=install_requires,
    long_description=long_description,
    long_description_content_type="text/markdown",
    extras_require=extras_require,
)
