import sys
import time
import re
import random
import datetime
from colorama import Fore, Style, init
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

init(autoreset=True)

# ----------------- Visual Animation Helpers -----------------
def show_typing_indicator(duration=1.2):
    """Simulates Misty thinking/typing with an animated dot indicator."""
    frames = ["Misty is typing .  ", "Misty is typing .. ", "Misty is typing ..."]
    end_time = time.time() + duration
    i = 0
    while time.time() < end_time:
        sys.stdout.write(f"\r{Fore.LIGHTBLACK_EX}{frames[i % len(frames)]}{Style.RESET_ALL}")
        sys.stdout.flush()
        time.sleep(0.3)
        i += 1
    # Clear the typing animation line completely
    sys.stdout.write("\r" + " " * 30 + "\r")
    sys.stdout.flush()

def typewriter_print(prefix, text, speed=0.015):
    """Prints response character-by-character like a live chat window."""
    sys.stdout.write(prefix)
    sys.stdout.flush()
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(speed)
    print("\n")

# ----------------- Knowledge Base & Intents -----------------
INTENT_DATA = {
    "greeting": {
        "patterns": [
            "hi", "hello", "hey", "good morning", "good evening", 
            "hey there", "what's up", "is anyone there"
        ],
        "responses": [
            "Hello! How can I assist you with your shopping experience today?",
            "Hi there! What can I help you resolve today?",
            "Greetings! How may I be of service?"
        ]
    },
    "identity": {
        "patterns": [
            "who are you", "who are u", "who r u", "what is your name", 
            "tell me about yourself", "are you a bot", "what do you do", 
            "introduce yourself"
        ],
        "responses": [
            "I am Misty, your 24/7 automated customer service assistant!",
            "I'm Misty, an AI assistant dedicated to helping with orders, returns, and support."
        ]
    },
    "services": {
        "patterns": [
            "what services do you provide", "what can you do", "help me with options",
            "what support do you offer", "list your features"
        ],
        "responses": [
            "I can assist you with:\n  * Tracking order shipments\n  * Initiating returns & refunds\n  * Downloading invoices\n  * Connecting you to human support"
        ]
    },
    "order_tracking": {
        "patterns": [
            "track my order", "where is my order", "order status", "track package",
            "check shipment", "where is my parcel", "delivery date"
        ],
        "responses": [
            "You can track your parcel by entering your Order ID (e.g., 'TRACK 12345') or via our website under the 'My Orders' section."
        ]
    },
    "return_policy": {
        "patterns": [
            "how to return product", "return policy", "replace item",
            "refund process", "i want to return", "exchange item", "money back"
        ],
        "responses": [
            "Our return policy allows hassle-free returns within 7 days of delivery directly from your account dashboard."
        ]
    },
    "invoice": {
        "patterns": [
            "how to get invoice", "download invoice", "bill copy",
            "tax invoice receipt", "need my receipt", "billing document"
        ],
        "responses": [
            "To download your invoice: Go to Account -> Orders -> Select Order -> Click 'Download Invoice (PDF)'."
        ]
    },
    "contact_human": {
        "patterns": [
            "contact support", "talk to human", "customer care number",
            "speak to agent", "email address", "call support", "live executive"
        ],
        "responses": [
            "You can reach our human support team directly:\n  * Email: support@example.com\n  * Toll-Free: 1800-123-456 (9 AM - 8 PM IST)"
        ]
    },
    "thanks": {
        "patterns": [
            "thank you", "thanks", "thanks a lot", "appreciate it", "great help"
        ],
        "responses": [
            "You're very welcome! Let me know if there's anything else.",
            "Glad I could help! Have a wonderful day ahead!",
            "Always happy to assist you!"
        ]
    },
    "goodbye": {
        "patterns": [
            "bye", "goodbye", "see you", "exit", "quit", "close chat"
        ],
        "responses": [
            "Goodbye! Thank you for contacting customer support.",
            "Take care! Feel free to reach out anytime.",
            "Have a great day ahead! Visit us again soon."
        ]
    }
}

# ----------------- NLP Matcher Engine -----------------
class NLPCustomerBot:
    def __init__(self, intents):
        self.intents = intents
        self.corpus = []
        self.intent_labels = []
        self._build_vectorizer()

    def _build_vectorizer(self):
        for intent, data in self.intents.items():
            for pattern in data["patterns"]:
                self.corpus.append(pattern.lower())
                self.intent_labels.append(intent)
                
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            token_pattern=r"(?u)\b\w+\b"
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)

    def _clean_text(self, text):
        return re.sub(r"[^\w\s]", "", text.lower()).strip()

    def handle_custom_actions(self, user_text):
        order_match = re.search(r"(?:track|order|id)\s*#?([a-zA-Z0-9]{4,8})", user_text, re.IGNORECASE)
        if order_match:
            order_id = order_match.group(1).upper()
            statuses = ["Out for Delivery", "In Transit", "Dispatched from Hub", "Delivered Yesterday"]
            chosen_status = random.choice(statuses)
            delivery_date = (datetime.date.today() + datetime.timedelta(days=2)).strftime("%b %d, %Y")
            return (
                f"\n{Fore.GREEN}📦 [Live Order Lookup]{Style.RESET_ALL}\n"
                f"  * Order ID : #{order_id}\n"
                f"  * Status   : {Fore.YELLOW}{chosen_status}{Style.RESET_ALL}\n"
                f"  * Est. Date: {delivery_date}"
            )
        return None

    def get_response(self, user_input):
        cleaned = self._clean_text(user_input)
        if not cleaned:
            return "Could you please type a valid query?", "neutral"

        action_output = self.handle_custom_actions(user_input)
        if action_output:
            return action_output, "tracking"

        for intent, data in self.intents.items():
            for pattern in data["patterns"]:
                if cleaned == pattern.lower() or pattern.lower() in cleaned:
                    return random.choice(data["responses"]), intent

        query_vec = self.vectorizer.transform([cleaned])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        best_idx = similarities.argmax()
        best_score = similarities[best_idx]

        if best_score > 0.20:
            matched_intent = self.intent_labels[best_idx]
            response = random.choice(self.intents[matched_intent]["responses"])
            return response, matched_intent

        return (
            "I'm not quite sure I understand that. Would you like to check order status, "
            "request a return, or talk to human support?",
            "fallback"
        )

# ----------------- Interactive Terminal Loop -----------------
def run_bot():
    bot = NLPCustomerBot(INTENT_DATA)
    
    print(f"\n{Fore.CYAN}=====================================================")
    print(f"{Fore.GREEN}        Misty - Customer Service Assistant v2.0")
    print(f"{Fore.CYAN}====================================================={Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Tip: Type 'bye', 'exit', or 'quit' to end the session.")
    print(f"Tip: Try typing 'where is my package' or 'track ID 84920'.{Style.RESET_ALL}\n")

    while True:
        try:
            user_msg = input(f"{Fore.BLUE}You: {Style.RESET_ALL}").strip()
            if not user_msg:
                continue

            response, intent = bot.get_response(user_msg)

            # Simulated natural delay with animated dots (1.0 to 1.5 seconds)
            show_typing_indicator(duration=random.uniform(0.9, 1.4))

            # Stream out the text character by character
            typewriter_print(f"{Fore.MAGENTA}Misty: {Style.RESET_ALL}", response, speed=0.015)

            if intent == "goodbye":
                break
        except (KeyboardInterrupt, EOFError):
            print(f"\n{Fore.MAGENTA}Misty: {Style.RESET_ALL}Session ended. Have a great day!")
            break

if __name__ == "__main__":
    run_bot()