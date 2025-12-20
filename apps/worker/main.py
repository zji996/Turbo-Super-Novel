from libs.pycore.paths import paths_summary


def main() -> None:
    summary = paths_summary()
    print(summary)
    print(
        "\nTo run Celery worker:\n"
        "  uv run --project apps/worker --directory apps/worker celery -A celery_app:celery_app worker -l info\n"
    )


if __name__ == "__main__":
    main()
