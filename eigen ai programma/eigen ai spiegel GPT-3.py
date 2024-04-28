import tkinter as tk
import openai

# Verbinden met de OpenAI API (vervang 'API_KEY' door je eigen API-sleutel)
openai.api_key = 'API_KEY'

def get_gpt3_response(prompt):
    # Gebruik GPT-3 om een reactie te genereren op basis van de prompt
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        temperature=0.7,
        max_tokens=50
    )
    return response.choices[0].text.strip()

def ask_question():
    question = question_entry.get()
    if question:
        # Voeg de vraag toe aan de chatgeschiedenis
        chat_history.insert(tk.END, f"Jij: {question}\n")
        chat_history.insert(tk.END, "AI: " + get_gpt3_response(question) + "\n\n")
        chat_history.see(tk.END)  # Scroll naar het laatste bericht
    else:
        chat_history.insert(tk.END, "Voer een vraag in.\n")

# GUI opzetten
root = tk.Tk()
root.title("Chat met AI")

chat_history = tk.Text(root, width=50, height=20)
chat_history.pack(padx=10, pady=10)

question_label = tk.Label(root, text="Vraag:")
question_label.pack()

question_entry = tk.Entry(root, width=50)
question_entry.pack(padx=10, pady=5)

ask_button = tk.Button(root, text="Stel vraag", command=ask_question)
ask_button.pack(pady=5)

root.mainloop()
