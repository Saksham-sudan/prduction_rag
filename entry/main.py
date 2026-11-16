from src.llm_call import call_func

while True:
    user_input = input("User> ")
    response = call_func(user_input)
    print(f"AI> {response[0].text}")