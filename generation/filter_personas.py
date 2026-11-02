import json
import fasttext

# Load the FastText language detection model (Make sure lid.176.bin is in the same directory)
model = fasttext.load_model("lid.176.bin")

# Read and filter English personas
filtered_personas = []
with open("persona.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        persona_text = next(iter(json.loads(line).values()))  # Extract only the text part
        lang_prediction = model.predict(persona_text, k=1)  # Predict language
        language = lang_prediction[0][0].replace("__label__", "")  # Extract language code

        if language == "en":  # Keep only English personas
            filtered_personas.append({"persona": persona_text})

# Save filtered personas back to a file
with open("filtered_personas.jsonl", "w", encoding="utf-8") as outfile:
    for persona in filtered_personas:
        outfile.write(json.dumps(persona) + "\n")

print(f"Filtered {len(filtered_personas)} English personas and saved to 'filtered_personas.jsonl'.")
