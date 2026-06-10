"""kyma-dub-enhanced — Neural Dubbing & Voice Synthesis Platform"""
from setuptools import setup, find_packages
from pathlib import Path

readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="kyma-dub-enhanced",
    version="1.0.0",
    description="Autonomous neural dubbing platform with voice cloning and lip-sync",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="kyma-dub-enhanced contributors",
    license="MIT",
    url="https://github.com/your-org/kyma-dub-enhanced",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "click>=8.1.7",
        "pyyaml>=6.0",
        "fastapi>=0.111.0",
        "uvicorn[standard]>=0.29.0",
        "pydantic>=2.7.0",
        "aiohttp>=3.9.5",
        "python-dotenv>=1.0.1",
        "apscheduler>=3.10.4",
        "openai-whisper>=20231117",
        "faster-whisper>=1.0.0",
        "transformers>=4.41.0",
        "torch>=2.3.0",
        "sentencepiece>=0.2.0",
        "deep-translator>=1.11.4",
        "TTS>=0.22.0",
        "bark>=0.1.5",
        "pyttsx3>=2.90",
        "soundfile>=0.12.1",
        "scipy>=1.13.0",
        "librosa>=0.10.2",
        "numpy>=1.26.4",
        "opencv-python-headless>=4.9.0",
        "sentence-transformers>=3.0.1",
        "pytest>=8.2.0",
        "pytest-asyncio>=0.23.7",
        "prometheus-client>=0.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.2.0",
            "pytest-asyncio>=0.23.7",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "kyma-dub=agent.main:cli",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",
    keywords="dubbing tts voice-cloning lip-sync whisper neural",
)
