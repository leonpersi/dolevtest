
from sentence_transformers import SentenceTransformer, util
import json

running = True

stories = [
    {"feeling": "sadness", "what_helped": "writing in a journal"},
    {"feeling": "loneliness", "what_helped": "talking to a close friend"},
    {"feeling": "stress", "what_helped": "taking a short walk and breathing slowly"},
    {"feeling": "anxiety", "what_helped": "focusing on the breath and grounding yourself"},
]

model = SentenceTransformer('all-MiniLM-L6-v2')

if running == True:
   while True:
        user_input = input("How are you feeling today? ")

        user_vec = model.encode(user_input, convert_to_tensor=True)
        story_vecs = model.encode([s["feeling"] for s in stories], convert_to_tensor=True)

        scores = util.cos_sim(user_vec, story_vecs)
        best_index = scores.argmax().item()

        match = stories[best_index]
        print("\nYou're not alone.")
        print("Others who felt", match["feeling"], "found that", match["what_helped"], "helped them.")
else:
    print("error")   