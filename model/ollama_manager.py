"""
Ollama Model Manager for GraphRAG

Handles automatic Ollama server management, model pulling, and health monitoring.
"""

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
from concurrent.futures import ThreadPoolExecutor, TimeoutError as ConcurrentTimeoutError

from .exceptions import OllamaError, TimeoutError, ModelNotFoundError, ServerStartError


class ModelStatus(Enum):
    """Status of an Ollama model."""
    NOT_INSTALLED = "not_installed"
    INSTALLING = "installing"
    INSTALLED = "installed"
    ERROR = "error"


@dataclass
class ModelInfo:
    """Information about an Ollama model."""
    name: str
    status: ModelStatus
    size: Optional[str] = None
    modified: Optional[str] = None
    error: Optional[str] = None


class OllamaManager:
    """
    Comprehensive Ollama model manager for GraphRAG.

    Features:
    - Auto-start Ollama server with timeout protection
    - Auto-pull required models (mistral, nomic-embed-text)
    - Comprehensive logging
    - Health monitoring
    - Graceful shutdown
    """

    # Required models for GraphRAG
    REQUIRED_MODELS = ["mistral", "nomic-embed-text"]

    # Ollama API endpoints
    OLLAMA_HOST = "http://localhost:11434"
    API_TAGS = f"{OLLAMA_HOST}/api/tags"
    API_PULL = f"{OLLAMA_HOST}/api/pull"
    API_SHOW = f"{OLLAMA_HOST}/api/show"

    # Timeouts (in seconds)
    SERVER_START_TIMEOUT = 60
    MODEL_PULL_TIMEOUT = 600  # 10 minutes
    HEALTH_CHECK_TIMEOUT = 10
    API_REQUEST_TIMEOUT = 30

    def __init__(self, log_dir: Optional[Path] = None):
        """
        Initialize Ollama manager.

        Args:
            log_dir: Directory to store Ollama logs. Defaults to model/ directory.
        """
        self.project_root = Path(__file__).parent.parent
        self.model_dir = self.project_root / "model"
        self.log_dir = log_dir or self.model_dir
        self.log_dir.mkdir(exist_ok=True)

        # Setup logging
        self.logger = self._setup_logging()

        # Server process
        self.server_process: Optional[subprocess.Popen] = None
        self.server_started = False

        # Model status cache
        self.model_cache: Dict[str, ModelInfo] = {}

        self.logger.info("🧠 Ollama Manager initialized")
        self.logger.info(f"📁 Log directory: {self.log_dir}")
        self.logger.info(f"🔗 Ollama API: {self.OLLAMA_HOST}")

    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging for Ollama operations."""
        logger = logging.getLogger("ollama_manager")
        logger.setLevel(logging.DEBUG)

        # Remove any existing handlers
        logger.handlers.clear()

        # File handler for all logs
        log_file = self.log_dir / "ollama.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)

        # Console handler for info and above
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def _run_with_timeout(self, func, timeout: int, *args, **kwargs):
        """Run a function with timeout protection."""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=timeout)
            except ConcurrentTimeoutError:
                self.logger.error(f"⏰ Operation timed out after {timeout} seconds")
                raise TimeoutError(f"Operation timed out after {timeout} seconds")

    def _check_ollama_binary(self) -> bool:
        """Check if Ollama binary is available."""
        try:
            result = subprocess.run(
                ["ollama", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                self.logger.info(f"✅ Ollama binary found: {version}")
                return True
            else:
                self.logger.error("❌ Ollama binary not found or not working")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.logger.error("❌ Ollama binary not found in PATH")
            return False

    def _is_server_running(self) -> bool:
        """Check if Ollama server is running."""
        try:
            response = requests.get(
                f"{self.OLLAMA_HOST}/api/tags",
                timeout=self.HEALTH_CHECK_TIMEOUT
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    def _start_ollama_server(self) -> bool:
        """Start Ollama server with timeout protection."""
        if self._is_server_running():
            self.logger.info("✅ Ollama server already running")
            self.server_started = True
            return True

        self.logger.info("🚀 Starting Ollama server...")

        try:
            # Start server in background
            ollama_log = self.log_dir / "ollama_server.log"
            with open(ollama_log, 'a') as log_file:
                self.server_process = subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid if os.name != 'nt' else None
                )

            # Wait for server to start with timeout
            start_time = time.time()
            while time.time() - start_time < self.SERVER_START_TIMEOUT:
                if self._is_server_running():
                    self.server_started = True
                    self.logger.info("✅ Ollama server started successfully")
                    return True
                time.sleep(1)

            # Timeout reached
            self._stop_server()
            raise ServerStartError("Ollama server failed to start within timeout")

        except FileNotFoundError:
            raise ServerStartError("Ollama binary not found")
        except Exception as e:
            self._stop_server()
            raise ServerStartError(f"Failed to start Ollama server: {e}")

    def _stop_server(self):
        """Stop Ollama server gracefully."""
        if self.server_process:
            self.logger.info("🛑 Stopping Ollama server...")
            try:
                if os.name == 'nt':
                    self.server_process.terminate()
                else:
                    os.killpg(os.getpgid(self.server_process.pid), signal.SIGTERM)

                # Wait for process to terminate
                try:
                    self.server_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.logger.warning("⚠️  Force killing Ollama server")
                    self.server_process.kill()

                self.logger.info("✅ Ollama server stopped")
            except Exception as e:
                self.logger.error(f"❌ Error stopping Ollama server: {e}")
            finally:
                self.server_process = None
                self.server_started = False

    def _get_installed_models(self) -> List[Dict]:
        """Get list of installed models from Ollama API."""
        try:
            response = requests.get(
                self.API_TAGS,
                timeout=self.API_REQUEST_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
            return data.get("models", [])
        except requests.RequestException as e:
            self.logger.error(f"❌ Failed to get installed models: {e}")
            return []

    def _pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry."""
        self.logger.info(f"📥 Pulling model: {model_name}")

        try:
            response = requests.post(
                self.API_PULL,
                json={"name": model_name},
                timeout=self.MODEL_PULL_TIMEOUT
            )
            response.raise_for_status()

            self.logger.info(f"✅ Successfully pulled model: {model_name}")
            return True

        except requests.Timeout:
            self.logger.error(f"⏰ Timeout pulling model: {model_name}")
            raise TimeoutError(f"Timeout pulling model {model_name}")
        except requests.RequestException as e:
            self.logger.error(f"❌ Failed to pull model {model_name}: {e}")
            raise OllamaError(f"Failed to pull model {model_name}: {e}")

    def _check_model_status(self, model_name: str) -> ModelInfo:
        """Check the status of a specific model."""
        installed_models = self._get_installed_models()
        model_names = [model["name"].split(":")[0] for model in installed_models]

        if model_name in model_names:
            # Get detailed info
            try:
                response = requests.post(
                    self.API_SHOW,
                    json={"name": model_name},
                    timeout=self.API_REQUEST_TIMEOUT
                )
                response.raise_for_status()
                data = response.json()

                return ModelInfo(
                    name=model_name,
                    status=ModelStatus.INSTALLED,
                    size=data.get("size"),
                    modified=data.get("modified_at")
                )
            except requests.RequestException:
                return ModelInfo(
                    name=model_name,
                    status=ModelStatus.ERROR,
                    error="Failed to get model details"
                )
        else:
            return ModelInfo(
                name=model_name,
                status=ModelStatus.NOT_INSTALLED
            )

    def ensure_models_ready(self) -> bool:
        """
        Ensure all required models are installed and ready.

        Returns:
            True if all models are ready, False otherwise
        """
        self.logger.info("🔍 Checking model readiness...")

        success = True
        for model_name in self.REQUIRED_MODELS:
            try:
                model_info = self._check_model_status(model_name)

                if model_info.status == ModelStatus.INSTALLED:
                    self.logger.info(f"✅ Model {model_name} is already installed")
                elif model_info.status == ModelStatus.NOT_INSTALLED:
                    self.logger.info(f"📥 Model {model_name} not found, pulling...")
                    self._pull_model(model_name)
                    self.logger.info(f"✅ Model {model_name} pulled successfully")
                else:
                    self.logger.error(f"❌ Model {model_name} has error: {model_info.error}")
                    success = False

            except Exception as e:
                self.logger.error(f"❌ Failed to ensure model {model_name}: {e}")
                success = False

        return success

    def get_model_status_report(self) -> str:
        """Generate a status report for all required models."""
        report_lines = ["🤖 Ollama Model Status Report", "=" * 40]

        for model_name in self.REQUIRED_MODELS:
            model_info = self._check_model_status(model_name)
            status_emoji = {
                ModelStatus.INSTALLED: "✅",
                ModelStatus.NOT_INSTALLED: "❌",
                ModelStatus.ERROR: "⚠️"
            }.get(model_info.status, "❓")

            report_lines.append(f"{status_emoji} {model_name}: {model_info.status.value}")

            if model_info.size:
                report_lines.append(f"   Size: {model_info.size}")
            if model_info.modified:
                report_lines.append(f"   Modified: {model_info.modified}")
            if model_info.error:
                report_lines.append(f"   Error: {model_info.error}")

        return "\n".join(report_lines)

    def initialize(self) -> bool:
        """
        Initialize Ollama environment - check binary, start server, ensure models.

        Returns:
            True if initialization successful, False otherwise
        """
        self.logger.info("🎯 Initializing Ollama environment...")

        try:
            # Step 1: Check Ollama binary
            if not self._check_ollama_binary():
                raise OllamaError("Ollama binary not available")

            # Step 2: Start server
            if not self._start_ollama_server():
                raise ServerStartError("Failed to start Ollama server")

            # Step 3: Ensure models are ready
            if not self.ensure_models_ready():
                raise ModelNotFoundError("Failed to ensure required models are available")

            self.logger.info("🎉 Ollama initialization completed successfully!")
            return True

        except Exception as e:
            self.logger.error(f"❌ Ollama initialization failed: {e}")
            return False

    def shutdown(self):
        """Shutdown Ollama manager and cleanup resources."""
        self.logger.info("🔄 Shutting down Ollama manager...")
        self._stop_server()
        self.logger.info("✅ Ollama manager shutdown complete")

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
