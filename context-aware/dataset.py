# dataset.py
# Tiny genre→movie mapping to simulate recommendations

def recommend(constraints):
    genre = constraints.get("genre", "")
    database = {
    "romance": [
        {"title": "The Notebook"}, {"title": "Pride & Prejudice"}, {"title": "Titanic"},
        {"title": "La La Land"}, {"title": "The Fault in Our Stars"}, {"title": "A Walk to Remember"},
        {"title": "Crazy Rich Asians"}, {"title": "The Vow"}, {"title": "About Time"},
        {"title": "The Holiday"}, {"title": "500 Days of Summer"}, {"title": "To All the Boys I've Loved Before"},
        {"title": "Love, Rosie"}, {"title": "Me Before You"}, {"title": "Dear John"},
        {"title": "Notting Hill"}, {"title": "The Proposal"}, {"title": "Runaway Bride"},
        {"title": "The Longest Ride"}, {"title": "Sleepless in Seattle"}, {"title": "When Harry Met Sally"},
        {"title": "Pretty Woman"}, {"title": "The Time Traveler’s Wife"}, {"title": "Letters to Juliet"},
        {"title": "The Age of Adaline"}, {"title": "Before Sunrise"}, {"title": "Before Sunset"},
        {"title": "Before Midnight"}, {"title": "The Last Song"}, {"title": "Safe Haven"}
    ],

    "horror": [
        {"title": "The Conjuring"}, {"title": "Get Out"}, {"title": "Hereditary"},
        {"title": "It"}, {"title": "The Exorcist"}, {"title": "A Quiet Place"},
        {"title": "Insidious"}, {"title": "Sinister"}, {"title": "The Babadook"},
        {"title": "Midsommar"}, {"title": "The Ring"}, {"title": "The Grudge"},
        {"title": "Annabelle"}, {"title": "Us"}, {"title": "The Nun"},
        {"title": "Poltergeist"}, {"title": "The Blair Witch Project"}, {"title": "Paranormal Activity"},
        {"title": "Smile"}, {"title": "The Others"}, {"title": "Talk to Me"},
        {"title": "Saw"}, {"title": "The Cabin in the Woods"}, {"title": "Barbarian"},
        {"title": "The Boogeyman"}, {"title": "The Witch"}, {"title": "Evil Dead Rise"},
        {"title": "X"}, {"title": "It Follows"}, {"title": "Nope"}
    ],

    "sci-fi": [
        {"title": "Interstellar"}, {"title": "The Matrix"}, {"title": "Inception"},
        {"title": "Blade Runner 2049"}, {"title": "Arrival"}, {"title": "Dune"},
        {"title": "Tenet"}, {"title": "Ex Machina"}, {"title": "Gravity"},
        {"title": "Edge of Tomorrow"}, {"title": "The Martian"}, {"title": "Avatar"},
        {"title": "Guardians of the Galaxy"}, {"title": "Star Wars: A New Hope"}, {"title": "Star Wars: The Empire Strikes Back"},
        {"title": "Star Wars: The Force Awakens"}, {"title": "Rogue One"}, {"title": "The Fifth Element"},
        {"title": "District 9"}, {"title": "Elysium"}, {"title": "Oblivion"},
        {"title": "I, Robot"}, {"title": "Minority Report"}, {"title": "Looper"},
        {"title": "Her"}, {"title": "Moon"}, {"title": "2001: A Space Odyssey"},
        {"title": "Jurassic Park"}, {"title": "War for the Planet of the Apes"}, {"title": "Ready Player One"}
    ],

    "action": [
        {"title": "John Wick"}, {"title": "Die Hard"}, {"title": "Mad Max: Fury Road"},
        {"title": "Gladiator"}, {"title": "The Dark Knight"}, {"title": "Inception"},
        {"title": "The Avengers"}, {"title": "Avengers: Endgame"}, {"title": "Iron Man"},
        {"title": "Black Panther"}, {"title": "The Batman"}, {"title": "Mission Impossible: Fallout"},
        {"title": "Top Gun: Maverick"}, {"title": "The Bourne Identity"}, {"title": "Skyfall"},
        {"title": "Casino Royale"}, {"title": "Taken"}, {"title": "Atomic Blonde"},
        {"title": "Extraction"}, {"title": "The Equalizer"}, {"title": "Man on Fire"},
        {"title": "The Raid"}, {"title": "Kill Bill: Vol. 1"}, {"title": "300"},
        {"title": "Deadpool"}, {"title": "Logan"}, {"title": "The Wolverine"},
        {"title": "Spider-Man: No Way Home"}, {"title": "Doctor Strange"}, {"title": "Shang-Chi"}
    ],

    "comedy": [
        {"title": "Superbad"}, {"title": "The Hangover"}, {"title": "Step Brothers"},
        {"title": "Bridesmaids"}, {"title": "Mean Girls"}, {"title": "Crazy, Stupid, Love"},
        {"title": "Tropic Thunder"}, {"title": "The 40-Year-Old Virgin"}, {"title": "Anchorman"},
        {"title": "Dumb and Dumber"}, {"title": "School of Rock"}, {"title": "Pitch Perfect"},
        {"title": "Zombieland"}, {"title": "Ted"}, {"title": "Knocked Up"},
        {"title": "The Other Guys"}, {"title": "21 Jump Street"}, {"title": "22 Jump Street"},
        {"title": "The Nice Guys"}, {"title": "Game Night"}, {"title": "Horrible Bosses"},
        {"title": "Rush Hour"}, {"title": "Hot Fuzz"}, {"title": "Shaun of the Dead"},
        {"title": "Juno"}, {"title": "The Intern"}, {"title": "Yes Man"},
        {"title": "Liar Liar"}, {"title": "The Mask"}, {"title": "Barb and Star Go to Vista Del Mar"}
    ],

    "drama": [
        {"title": "The Shawshank Redemption"}, {"title": "Forrest Gump"}, {"title": "Fight Club"},
        {"title": "The Godfather"}, {"title": "The Green Mile"}, {"title": "The Pursuit of Happyness"},
        {"title": "A Beautiful Mind"}, {"title": "The Social Network"}, {"title": "Good Will Hunting"},
        {"title": "Whiplash"}, {"title": "Requiem for a Dream"}, {"title": "The Pianist"},
        {"title": "The Imitation Game"}, {"title": "Parasite"}, {"title": "Everything Everywhere All at Once"},
        {"title": "12 Years a Slave"}, {"title": "The Revenant"}, {"title": "Joker"},
        {"title": "Black Swan"}, {"title": "Birdman"}, {"title": "Oppenheimer"},
        {"title": "The Whale"}, {"title": "The Father"}, {"title": "Moonlight"},
        {"title": "Manchester by the Sea"}, {"title": "Revolutionary Road"}, {"title": "The Great Gatsby"},
        {"title": "American Beauty"}, {"title": "Marriage Story"}, {"title": "Bohemian Rhapsody"}
    ],

    "thriller": [
        {"title": "Se7en"}, {"title": "Gone Girl"}, {"title": "Shutter Island"},
        {"title": "Prisoners"}, {"title": "The Girl with the Dragon Tattoo"}, {"title": "Fight Club"},
        {"title": "Nightcrawler"}, {"title": "The Prestige"}, {"title": "Memento"},
        {"title": "Zodiac"}, {"title": "The Sixth Sense"}, {"title": "Split"},
        {"title": "Don’t Breathe"}, {"title": "Phone Booth"}, {"title": "Enemy"},
        {"title": "No Country for Old Men"}, {"title": "Wind River"}, {"title": "Sicario"},
        {"title": "Collateral"}, {"title": "Training Day"}, {"title": "Run Lola Run"},
        {"title": "Oldboy"}, {"title": "Gone Baby Gone"}, {"title": "The Invisible Man"},
        {"title": "The Game"}, {"title": "Tenet"}, {"title": "Heat"},
        {"title": "The Fugitive"}, {"title": "Panic Room"}, {"title": "The Machinist"}
    ],

    "fantasy": [
        {"title": "Harry Potter and the Sorcerer’s Stone"}, {"title": "The Lord of the Rings: The Fellowship of the Ring"},
        {"title": "The Lord of the Rings: The Two Towers"}, {"title": "The Lord of the Rings: The Return of the King"},
        {"title": "The Hobbit: An Unexpected Journey"}, {"title": "Fantastic Beasts and Where to Find Them"},
        {"title": "Pan’s Labyrinth"}, {"title": "The Chronicles of Narnia"}, {"title": "Percy Jackson & the Olympians"},
        {"title": "Stardust"}, {"title": "Alice in Wonderland"}, {"title": "Maleficent"},
        {"title": "Snow White and the Huntsman"}, {"title": "The Golden Compass"}, {"title": "Eragon"},
        {"title": "Doctor Strange"}, {"title": "Thor: Ragnarok"}, {"title": "Wonder Woman"},
        {"title": "Aquaman"}, {"title": "The Shape of Water"}, {"title": "Big Fish"},
        {"title": "Bridge to Terabithia"}, {"title": "Enchanted"}, {"title": "Mirror Mirror"},
        {"title": "The Spiderwick Chronicles"}, {"title": "Miss Peregrine’s Home for Peculiar Children"},
        {"title": "The Princess Bride"}, {"title": "Peter Pan"}, {"title": "Hook"}, {"title": "The NeverEnding Story"}
    ],

    "animation": [
        {"title": "Toy Story"}, {"title": "Toy Story 3"}, {"title": "Finding Nemo"},
        {"title": "The Incredibles"}, {"title": "Up"}, {"title": "Inside Out"},
        {"title": "Coco"}, {"title": "Soul"}, {"title": "Monsters, Inc."},
        {"title": "Ratatouille"}, {"title": "WALL·E"}, {"title": "Frozen"},
        {"title": "Moana"}, {"title": "Zootopia"}, {"title": "Encanto"},
        {"title": "Turning Red"}, {"title": "Shrek"}, {"title": "Shrek 2"},
        {"title": "Kung Fu Panda"}, {"title": "How to Train Your Dragon"},
        {"title": "The Lion King"}, {"title": "Beauty and the Beast"}, {"title": "Aladdin"},
        {"title": "Spirited Away"}, {"title": "My Neighbor Totoro"}, {"title": "Kiki’s Delivery Service"},
        {"title": "Your Name"}, {"title": "Weathering With You"}, {"title": "The Mitchells vs. The Machines"}, {"title": "Big Hero 6"}
    ],

    "crime": [
        {"title": "The Godfather"}, {"title": "The Godfather Part II"}, {"title": "Goodfellas"},
        {"title": "The Irishman"}, {"title": "Casino"}, {"title": "The Departed"},
        {"title": "Scarface"}, {"title": "Heat"}, {"title": "American Gangster"},
        {"title": "City of God"}, {"title": "The Untouchables"}, {"title": "Training Day"},
        {"title": "Pulp Fiction"}, {"title": "Reservoir Dogs"}, {"title": "Snatch"},
        {"title": "Lock, Stock and Two Smoking Barrels"}, {"title": "The Town"}, {"title": "The Usual Suspects"},
        {"title": "The Wolf of Wall Street"}, {"title": "Catch Me If You Can"}, {"title": "Blow"},
        {"title": "Donnie Brasco"}, {"title": "Sicario"}, {"title": "Public Enemies"},
        {"title": "The French Connection"}, {"title": "Gangs of New York"}, {"title": "Zodiac"},
        {"title": "Mystic River"}, {"title": "Eastern Promises"}, {"title": "Black Mass"}
    ],

    "mystery": [
        {"title": "Knives Out"}, {"title": "Glass Onion"}, {"title": "Gone Girl"},
        {"title": "Shutter Island"}, {"title": "The Sixth Sense"}, {"title": "Prisoners"},
        {"title": "The Girl on the Train"}, {"title": "Murder on the Orient Express"},
        {"title": "Death on the Nile"}, {"title": "The Others"}, {"title": "Seven"},
        {"title": "Zodiac"}, {"title": "The Prestige"}, {"title": "The Illusionist"},
        {"title": "Mulholland Drive"}, {"title": "Mystic River"}, {"title": "Secret Window"},
        {"title": "Oldboy"}, {"title": "The Number 23"}, {"title": "Identity"},
        {"title": "Now You See Me"}, {"title": "The Da Vinci Code"}, {"title": "Angels & Demons"},
        {"title": "Sherlock Holmes"}, {"title": "Sherlock Holmes: A Game of Shadows"},
        {"title": "Enola Holmes"}, {"title": "The Invisible Guest"}, {"title": "Fracture"},
        {"title": "Enemy"}, {"title": "The Machinist"}
    ],

    "adventure": [
        {"title": "Indiana Jones and the Raiders of the Lost Ark"}, {"title": "Indiana Jones and the Last Crusade"},
        {"title": "Pirates of the Caribbean: The Curse of the Black Pearl"}, {"title": "Pirates of the Caribbean: Dead Man’s Chest"},
        {"title": "Jurassic Park"}, {"title": "The Lost World: Jurassic Park"}, {"title": "The Mummy"},
        {"title": "The Mummy Returns"}, {"title": "National Treasure"}, {"title": "National Treasure: Book of Secrets"},
        {"title": "Jumanji"}, {"title": "Jumanji: Welcome to the Jungle"}, {"title": "King Kong"},
        {"title": "The Jungle Book"}, {"title": "Life of Pi"}, {"title": "Avatar"},
        {"title": "The Revenant"}, {"title": "Cast Away"}, {"title": "The Secret Life of Walter Mitty"},
        {"title": "The Hobbit: The Desolation of Smaug"}, {"title": "The Hobbit: The Battle of the Five Armies"},
        {"title": "The Lord of the Rings: The Return of the King"}, {"title": "The Lion King"},
        {"title": "Up"}, {"title": "The Grand Budapest Hotel"}, {"title": "Interstellar"},
        {"title": "The Martian"}, {"title": "Kingdom of Heaven"}, {"title": "Tomb Raider"}, {"title": "Uncharted"}
    ],

    "biopic": [
        {"title": "The Social Network"}, {"title": "Steve Jobs"}, {"title": "The Imitation Game"},
        {"title": "A Beautiful Mind"}, {"title": "The Theory of Everything"}, {"title": "Bohemian Rhapsody"},
        {"title": "Rocketman"}, {"title": "Lincoln"}, {"title": "Schindler’s List"},
        {"title": "Catch Me If You Can"}, {"title": "The Pursuit of Happyness"}, {"title": "Oppenheimer"},
        {"title": "The Founder"}, {"title": "American Sniper"}, {"title": "The Wolf of Wall Street"},
        {"title": "Ray"}, {"title": "Walk the Line"}, {"title": "The King’s Speech"},
        {"title": "12 Years a Slave"}, {"title": "Moneyball"}, {"title": "Hidden Figures"},
        {"title": "The Blind Side"}, {"title": "Erin Brockovich"}, {"title": "Hacksaw Ridge"},
        {"title": "Into the Wild"}, {"title": "The Pianist"}, {"title": "The Disaster Artist"},
        {"title": "Blow"}, {"title": "Goodfellas"}, {"title": "The Irishman"}
    ],

    "musical": [
        {"title": "La La Land"}, {"title": "Mamma Mia!"}, {"title": "Mamma Mia! Here We Go Again"},
        {"title": "The Greatest Showman"}, {"title": "Chicago"}, {"title": "Les Misérables"},
        {"title": "Grease"}, {"title": "West Side Story"}, {"title": "Moulin Rouge!"},
        {"title": "Rocketman"}, {"title": "Bohemian Rhapsody"}, {"title": "A Star Is Born"},
        {"title": "Pitch Perfect"}, {"title": "Pitch Perfect 2"}, {"title": "Pitch Perfect 3"},
        {"title": "High School Musical"}, {"title": "Encanto"}, {"title": "Frozen"},
        {"title": "Beauty and the Beast"}, {"title": "The Little Mermaid"}, {"title": "Hairspray"},
        {"title": "Sing"}, {"title": "Sing 2"}, {"title": "Annie"}, {"title": "The Sound of Music"},
        {"title": "Dreamgirls"}, {"title": "Cinderella"}, {"title": "Into the Woods"},
        {"title": "Tick, Tick... Boom!"}, {"title": "Once"}
    ],

    "historical": [
        {"title": "Schindler’s List"}, {"title": "Gladiator"}, {"title": "Braveheart"},
        {"title": "Saving Private Ryan"}, {"title": "1917"}, {"title": "The Last Samurai"},
        {"title": "The Patriot"}, {"title": "The King’s Speech"}, {"title": "Dunkirk"},
        {"title": "The Imitation Game"}, {"title": "Oppenheimer"}, {"title": "Lincoln"},
        {"title": "12 Years a Slave"}, {"title": "The Pianist"}, {"title": "Hotel Rwanda"},
        {"title": "Gandhi"}, {"title": "Lawrence of Arabia"}, {"title": "The Bridge on the River Kwai"},
        {"title": "The Last Emperor"}, {"title": "Troy"}, {"title": "Kingdom of Heaven"},
        {"title": "Darkest Hour"}, {"title": "The Post"}, {"title": "Defiance"},
        {"title": "The Thin Red Line"}, {"title": "Apocalypto"}, {"title": "Enemy at the Gates"},
        {"title": "Pearl Harbor"}, {"title": "Elizabeth"}, {"title": "The Favourite"}
   
    ],
    }
    return database.get(genre, [{"title": "Inception"}])
