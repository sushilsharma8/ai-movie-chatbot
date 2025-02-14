from locust import HttpUser, task, between, events
import random

# Sample test data
CHARACTERS = ["Tony Stark", "Thor", "Black Widow", "Captain America", "Hulk"]
MESSAGES = [
    "How do I build better armor?",
    "What's the best way to defeat Thanos?",
    "Tell me about your superpowers.",
    "How do you handle stress?",
    "What's your favorite battle strategy?"
]

class ChatUser(HttpUser):
    wait_time = between(0.01, 0.1)  # Simulate high traffic

    @task
    def send_chat(self):
        # Randomly select character and message
        character = random.choice(CHARACTERS)
        message = random.choice(MESSAGES)
        
        # Send POST request to /chat endpoint
        self.client.post(
            "/chat",
            json={
                "character": character,
                "user_message": message
            },
            headers={"Content-Type": "application/json"}
        )

# Optional: Add custom metrics
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    if exception:
        print(f"Request failed: {name} - {exception}")
    else:
        print(f"Request succeeded: {name} - {response_time}ms")