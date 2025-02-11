import requests
from bs4 import BeautifulSoup
import re
from main import SessionLocal, MovieScript  # Import database models

# Base URL of IMSDb
IMSDb_URL = "https://imsdb.com"

# Correct Marvel movie script slugs
MARVEL_MOVIES = [
    "Avengers-Endgame",
    # "Captain-America-Civil-War",    
    # "Iron-Man-3",
    # "Thor-Ragnarok",
]

def clean_dialogue(dialogue):
    """
    Removes scene descriptions that accidentally get stored after dialogue.
    """
    # Remove text that looks like an action or direction after dialogue
    dialogue = re.split(r"(?<=\w\.)\s+[A-Z ]{3,30}\s", dialogue)[0]

    # Remove extra spaces
    return dialogue.strip()

def get_movie_script(movie_slug):
    """Scrape the script of a given Marvel movie from IMSDb and extract clean dialogues."""
    url = f"{IMSDb_URL}/scripts/{movie_slug}.html"
    print(f"Scraping {url} ...")

    response = requests.get(url)
    if response.status_code != 200:
        print(f"❌ Failed to fetch {movie_slug}")
        return None

    soup = BeautifulSoup(response.text, "lxml")

    # Extract script content (IMSDb stores it inside <pre> tags)
    script_text = soup.find("pre")
    if not script_text:
        print(f"❌ Could not find script text for {movie_slug}")
        return None

    script_lines = script_text.text.split("\n")

    dialogues = []
    current_character = None
    current_dialogue = ""

    for line in script_lines:
        line = line.strip()

        # Ignore empty lines
        if not line:
            continue

        # Detect character names (uppercase, reasonable length, no punctuation)
        if re.match(r"^[A-Z ]{3,30}$", line) and not re.search(r"\(.*\)|EXT\.|INT\.|FADE|CUT TO|DISSOLVE|MONTAGE|ANGLE ON|CLOSE ON|CONT'D", line, re.IGNORECASE):
            # If there's an ongoing dialogue, save it before switching to a new character
            if current_character and current_dialogue:
                dialogues.append((current_character, clean_dialogue(current_dialogue.strip())))
                current_dialogue = ""

            current_character = line.strip()  # Store new character name
        
        # Detect spoken dialogue and merge multi-line text
        elif (
            current_character and 
            not re.search(r"^\(.*\)$", line) and   # Ignore lines inside parentheses (scene descriptions)
            not re.search(r"EXT\.|INT\.|FADE|CUT TO|DISSOLVE|MONTAGE|ANGLE ON|CLOSE ON|CONT'D", line, re.IGNORECASE) and  # Ignore screenplay directions
            not re.match(r"^[A-Z ]{3,30}$", line)  # Ignore uppercase scene descriptions
        ):
            current_dialogue += " " + line.strip()

    # Save last character dialogue
    if current_character and current_dialogue:
        dialogues.append((current_character, clean_dialogue(current_dialogue.strip())))

    return dialogues

def save_to_database(movie, dialogues):
    """Store clean movie dialogues in NeonDB."""
    session = SessionLocal()
    
    for character, text in dialogues:
        new_script = MovieScript(character=character, dialogue=text)
        session.add(new_script)
    
    session.commit()
    session.close()
    print(f"✅ Saved {len(dialogues)} dialogues from {movie} to NeonDB.")

# Run the scraper for Marvel movies
for movie in MARVEL_MOVIES:
    dialogues = get_movie_script(movie)
    if dialogues:
        save_to_database(movie, dialogues)
