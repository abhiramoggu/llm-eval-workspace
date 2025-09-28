# user_sim.py
import random

class UserSimulator:
    def __init__(self):
        self.true_genre = random.choice(["romance", "horror", "sci-fi"])
        self.error_injected = False

    def get_message(self, turn: int) -> str:
        if turn == 0:
            # First turn: state preference, maybe with error
            if random.random() < 0.5:
                wrong_genre = random.choice(["romance", "horror", "sci-fi"])
                while wrong_genre == self.true_genre:
                    wrong_genre = random.choice(["romance", "horror", "sci-fi"])
                self.error_injected = True
                return f"I want to watch a {wrong_genre} movie."
            else:
                return f"I want to watch a {self.true_genre} movie."

        elif turn == 1 and self.error_injected:
            return f"Oops, I meant {self.true_genre}, not the other genre."

        else:
            return "Thanks for the recommendation!"
