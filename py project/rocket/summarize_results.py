from config import GA_CANDIDATE_FILE, RESULT_FILE, SUMMARY_FILE
from core import summarize_results


def main(result_file=None):
    if result_file is None:
        result_file = RESULT_FILE if RESULT_FILE.exists() else GA_CANDIDATE_FILE
    summarize_results(result_file, SUMMARY_FILE)


if __name__ == "__main__":
    main()
