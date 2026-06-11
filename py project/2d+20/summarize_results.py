from config import RESULT_FILE, SUMMARY_FILE
from core import summarize_results


def main():
    summarize_results(RESULT_FILE, SUMMARY_FILE)


if __name__ == "__main__":
    main()
