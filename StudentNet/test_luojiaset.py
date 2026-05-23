import sys
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1]))

from luojiaset_runners import BaseLuojiaTestScript  # noqa: E402


class StudentLuojiaTestScript(BaseLuojiaTestScript):
    report_mode = "joint"


if __name__ == "__main__":
    StudentLuojiaTestScript().run()
