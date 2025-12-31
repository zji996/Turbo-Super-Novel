from core.paths import paths_summary


def main() -> None:
    summary = paths_summary()
    print(summary)
    print(
        "\nTo run Celery worker:\n"
        "  WORKER_CAPABILITIES=tts,videogen CAP_GPU_MODE=resident \\\n"
        "  uv run --project apps/worker --directory apps/worker celery -A celery_app:celery_app worker -l info \\\n"
        "    -Q celery,cap.tts,cap.videogen --concurrency 1 --prefetch-multiplier 1\n"
    )


if __name__ == "__main__":
    main()
