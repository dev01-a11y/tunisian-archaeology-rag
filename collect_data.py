import wikipediaapi
import os
import time

wiki = wikipediaapi.Wikipedia(
    user_agent='TunisianArcheologyChatbot/1.0 (Educational Project)',
    language='en'
)

# NEW topics not in your collection
additional_topics = [
    # More Tunisian Sites
    "Bardo National Museum",
    "Gightis",
    "Althiburos",
    "Mustis",
    "Uchi Maius",
    "Thubursicu Numidarum",
    "Simitthu",
    "Thibaris"
]

os.makedirs('data/raw_documents', exist_ok=True)

print("Downloading additional topics...")
count = 0
for topic in additional_topics:
    try:
        page = wiki.page(topic)
        if page.exists():
            filename = f"data/raw_documents/extra_{topic.replace(' ', '_').lower()}_en.txt"
            
            if os.path.exists(filename):
                print(f"⊘ Already exists: {topic}")
                continue
                
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Title: {page.title}\n")
                f.write(f"Source: Wikipedia\n")
                f.write(f"Topic: {topic}\n")
                f.write(f"Category: Archaeological/Historical Reference\n\n")
                f.write(page.text)
            print(f"✓ Downloaded: {topic}")
            count += 1
        time.sleep(0.5)
    except Exception as e:
        print(f"✗ Error: {topic} - {e}")

print(f"\n✓ Additional topics downloaded: {count}")
