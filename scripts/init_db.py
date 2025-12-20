from __future__ import annotations

from libs.dbcore import create_all


def main() -> None:
    create_all()
    print("ok")


if __name__ == "__main__":
    main()

