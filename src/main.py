#!/usr/bin/env python3
"""
Main entry point for Vehicle Simulation.
Starts both Simulation API and Chat API.
"""
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import CHAT_API_PORT, QDRANT_HOST, QDRANT_PORT, SIM_API_PORT


@dataclass
class Service:
    """A managed service."""
    name: str
    command: list[str]
    port: int
    process: Optional[subprocess.Popen] = None

    def start(self) -> bool:
        """Start the service."""
        print(f"[START] {self.name} on port {self.port}...")

        try:
            self.process = subprocess.Popen(
                self.command,
                stdout=None,  # Print to console
                stderr=None,
            )

            time.sleep(1)

            if self.process.poll() is not None:
                print(f"[ERROR] {self.name} failed to start")
                return False

            print(f"[OK] {self.name} started (PID: {self.process.pid})")
            return True

        except Exception as e:
            print(f"[ERROR] {self.name}: {e}")
            return False

    def stop(self):
        """Stop the service."""
        if self.process and self.process.poll() is None:
            print(f"[STOP] {self.name}...")
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def is_running(self) -> bool:
        """Check if running."""
        return self.process and self.process.poll() is None


class ServiceManager:
    """Manages multiple services."""

    def __init__(self):
        self.services: list[Service] = []
        self._setup_signals()

    def _setup_signals(self):
        """Setup signal handlers."""
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, sig, frame):
        """Handle shutdown signal."""
        print("\n[SIGNAL] Shutting down...")
        self.stop_all()
        sys.exit(0)

    def add(self, service: Service):
        """Add a service."""
        self.services.append(service)

    def start_all(self) -> bool:
        """Start all services."""
        print("=" * 60)
        print("STARTING SERVICES")
        print("=" * 60)

        all_ok = True
        for service in self.services:
            if not service.start():
                all_ok = False

        return all_ok

    def stop_all(self):
        """Stop all services."""
        print("\n[SHUTDOWN] Stopping all services...")
        for service in reversed(self.services):
            service.stop()
        print("[SHUTDOWN] Done")

    def monitor(self):
        """Monitor services."""
        print("\n" + "=" * 60)
        print("SERVICES RUNNING")
        print("=" * 60)
        print("Press Ctrl+C to stop\n")

        try:
            while True:
                for service in self.services:
                    if not service.is_running():
                        print(f"[WARN] {service.name} stopped unexpectedly!")
                time.sleep(2)
        except KeyboardInterrupt:
            pass


def check_qdrant() -> bool:
    """Check if Qdrant is accessible."""
    print("[CHECK] Qdrant...")
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=3)
        client.get_collections()
        print("[OK] Qdrant connected")
        return True
    except Exception as e:
        print(f"[WARN] Qdrant not available: {e}")
        print("[INFO] Start with: docker compose up -d")
        return False


def main():
    """Main entry point."""
    print("=" * 60)
    print("VEHICLE SIMULATION STARTUP")
    print("=" * 60)

    # Check dependencies
    check_qdrant()

    # Create service manager
    manager = ServiceManager()

    # Add Simulation API
    manager.add(Service(
        name="Simulation API",
        command=[
            sys.executable, "-c",
            f"import uvicorn; from src.simulation.api import app; uvicorn.run(app, host='0.0.0.0', port={SIM_API_PORT})"
        ],
        port=SIM_API_PORT,
    ))

    # Add Chat API
    manager.add(Service(
        name="Chat API",
        command=[
            sys.executable, "-c",
            f"import uvicorn; from src.chat.app import app; uvicorn.run(app, host='0.0.0.0', port={CHAT_API_PORT})"
        ],
        port=CHAT_API_PORT,
    ))

    # Start services
    if not manager.start_all():
        print("\n[WARN] Some services failed to start")

    # Show access info
    print("\n" + "=" * 60)
    print("ACCESS POINTS")
    print("=" * 60)
    print(f"  Dashboard:       http://localhost:{CHAT_API_PORT}")
    print(f"  Simulation API:  http://localhost:{SIM_API_PORT}")
    print(f"  Chat API:        http://localhost:{CHAT_API_PORT}/chat")
    print("=" * 60 + "\n")

    # Monitor
    manager.monitor()

    # Cleanup
    manager.stop_all()


if __name__ == "__main__":
    main()
