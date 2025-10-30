# For MedGemma-4B using LMStudio
# from langchain_openai import ChatOpenAI   # ✅ use the community wrapper

# llm = ChatOpenAI(
#     model_name="medgemma-4b-it",
#     base_url="http://192.168.1.128:1234/v1",
#     api_key="not-needed", # The API key can be anything
#     temperature=0.1,
#     max_tokens=1024
# )
# result = llm.invoke("What is the capital of India?")
# print(result)



# For Ngrok using Collab fopr Medgemma-27B
# from langchain_openai import ChatOpenAI

# llm = ChatOpenAI(
#     model_name="medgemma-27b-it",
#     base_url="https://lancinate-persuasive-paxton.ngrok-free.dev/v1",  # your latest ngrok URL
#     api_key="not-needed",
#     temperature=0.5
# )

# response = llm.invoke("Explain COPD in simple terms.")
# print(response.content)



count_discharge = 0
count_history = 0
with open("data/MIMIC_NOTE.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.lower()
        if "discharge" in line:
            count_discharge += 1
        if "history of present illness" in line or "hospital course" in line:
            count_history += 1
print("Discharge:", count_discharge, "History/Hospital Course:", count_history)
