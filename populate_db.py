from main import SessionLocal, MovieScript

session = SessionLocal()

# Add sample dialogues
scripts = [
    MovieScript(character="Batman", dialogue="I am Batman!"),
    MovieScript(character="Joker", dialogue="Why so serious?"),
    MovieScript(character="Iron Man", dialogue="I am Iron Man."),
]

session.add_all(scripts)
session.commit()
session.close()

print("NeonDB populated successfully!")
