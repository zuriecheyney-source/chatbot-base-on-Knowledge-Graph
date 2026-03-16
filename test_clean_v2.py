from answer_search import AnswerSearcher

def test_clean():
    s = AnswerSearcher.__new__(AnswerSearcher) # Avoid DB connection for simple static test
    test_data = ["7天左右", "7天", "14天；", "14天", "7--14天", "2-4周", "治疗周期5-10天"]
    cleaned = s._clean_list(test_data)
    print(f"Original: {test_data}")
    print(f"Cleaned:  {cleaned}")

if __name__ == "__main__":
    test_clean()
