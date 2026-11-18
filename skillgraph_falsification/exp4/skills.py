"""Pre-registered lexical skill inventory for Experiments 2 (single skills) and 4 (pairs).

Each skill is a deterministic, checkable function with a typed domain, an English
noun-phrase template (so two skills compose into one natural question), and a set of
corpus cue phrases used to estimate how often the skill is explicitly exercised in
the Pile (infini-gram counts). Cue phrases are a crude, pre-registered proxy for the
doc's trained tagger; see PREREGISTRATION.md.
"""

# ----------------------------------------------------------------------------- data
COUNTRIES = {
    # country: (capital, language, continent, currency)
    "France": ("Paris", "French", "Europe", "euro"),
    "Germany": ("Berlin", "German", "Europe", "euro"),
    "Italy": ("Rome", "Italian", "Europe", "euro"),
    "Spain": ("Madrid", "Spanish", "Europe", "euro"),
    "Portugal": ("Lisbon", "Portuguese", "Europe", "euro"),
    "Greece": ("Athens", "Greek", "Europe", "euro"),
    "Netherlands": ("Amsterdam", "Dutch", "Europe", "euro"),
    "Austria": ("Vienna", "German", "Europe", "euro"),
    "Finland": ("Helsinki", "Finnish", "Europe", "euro"),
    "Sweden": ("Stockholm", "Swedish", "Europe", "krona"),
    "Norway": ("Oslo", "Norwegian", "Europe", "krone"),
    "Denmark": ("Copenhagen", "Danish", "Europe", "krone"),
    "Poland": ("Warsaw", "Polish", "Europe", "zloty"),
    "Hungary": ("Budapest", "Hungarian", "Europe", "forint"),
    "Japan": ("Tokyo", "Japanese", "Asia", "yen"),
    "China": ("Beijing", "Chinese", "Asia", "yuan"),
    "Thailand": ("Bangkok", "Thai", "Asia", "baht"),
    "Vietnam": ("Hanoi", "Vietnamese", "Asia", "dong"),
    "South Korea": ("Seoul", "Korean", "Asia", "won"),
    "Indonesia": ("Jakarta", "Indonesian", "Asia", "rupiah"),
    "Iran": ("Tehran", "Persian", "Asia", "rial"),
    "Saudi Arabia": ("Riyadh", "Arabic", "Asia", "riyal"),
    "Egypt": ("Cairo", "Arabic", "Africa", "pound"),
    "Morocco": ("Rabat", "Arabic", "Africa", "dirham"),
    "Ethiopia": ("Addis Ababa", "Amharic", "Africa", "birr"),
    "Nigeria": ("Abuja", "English", "Africa", "naira"),
    "Brazil": ("Brasilia", "Portuguese", "South America", "real"),
    "Argentina": ("Buenos Aires", "Spanish", "South America", "peso"),
    "Peru": ("Lima", "Spanish", "South America", "sol"),
    "Chile": ("Santiago", "Spanish", "South America", "peso"),
    "Colombia": ("Bogota", "Spanish", "South America", "peso"),
    "Mexico": ("Mexico City", "Spanish", "North America", "peso"),
    "Cuba": ("Havana", "Spanish", "North America", "peso"),
}

ADJ = {
    # adj: (opposite, comparative, french)
    "hot": ("cold", "hotter", "chaud"),
    "cold": ("hot", "colder", "froid"),
    "big": ("small", "bigger", "grand"),
    "small": ("big", "smaller", "petit"),
    "fast": ("slow", "faster", "rapide"),
    "slow": ("fast", "slower", "lent"),
    "happy": ("sad", "happier", "heureux"),
    "sad": ("happy", "sadder", "triste"),
    "good": ("bad", "better", "bon"),
    "bad": ("good", "worse", "mauvais"),
    "rich": ("poor", "richer", "riche"),
    "poor": ("rich", "poorer", "pauvre"),
    "strong": ("weak", "stronger", "fort"),
    "weak": ("strong", "weaker", "faible"),
    "full": ("empty", "fuller", "plein"),
    "empty": ("full", "emptier", "vide"),
    "clean": ("dirty", "cleaner", "propre"),
    "dirty": ("clean", "dirtier", "sale"),
    "high": ("low", "higher", "haut"),
    "low": ("high", "lower", "bas"),
    "heavy": ("light", "heavier", "lourd"),
    "light": ("heavy", "lighter", "léger"),
    "wet": ("dry", "wetter", "mouillé"),
    "dry": ("wet", "drier", "sec"),
    "long": ("short", "longer", "long"),
    "short": ("long", "shorter", "court"),
    "thick": ("thin", "thicker", "épais"),
    "thin": ("thick", "thinner", "mince"),
    "wide": ("narrow", "wider", "large"),
    "narrow": ("wide", "narrower", "étroit"),
}

VERB = {
    # verb: (opposite, past)
    "win": ("lose", "won"), "lose": ("win", "lost"),
    "buy": ("sell", "bought"), "sell": ("buy", "sold"),
    "give": ("take", "gave"), "take": ("give", "took"),
    "come": ("go", "came"), "go": ("come", "went"),
    "open": ("close", "opened"), "close": ("open", "closed"),
    "push": ("pull", "pushed"), "pull": ("push", "pulled"),
    "start": ("stop", "started"), "stop": ("start", "stopped"),
    "love": ("hate", "loved"), "hate": ("love", "hated"),
    "arrive": ("leave", "arrived"), "leave": ("arrive", "left"),
    "remember": ("forget", "remembered"), "forget": ("remember", "forgot"),
    "accept": ("refuse", "accepted"), "refuse": ("accept", "refused"),
    "lend": ("borrow", "lent"), "borrow": ("lend", "borrowed"),
    "send": ("receive", "sent"), "receive": ("send", "received"),
    "sit": ("stand", "sat"), "stand": ("sit", "stood"),
    "rise": ("fall", "rose"), "fall": ("rise", "fell"),
    "increase": ("decrease", "increased"), "decrease": ("increase", "decreased"),
}

ANIMAL = {
    # animal: (baby, plural of baby)
    "dog": ("puppy", "puppies"), "cat": ("kitten", "kittens"), "cow": ("calf", "calves"),
    "sheep": ("lamb", "lambs"), "pig": ("piglet", "piglets"), "duck": ("duckling", "ducklings"),
    "frog": ("tadpole", "tadpoles"), "bear": ("cub", "cubs"), "deer": ("fawn", "fawns"),
    "goose": ("gosling", "goslings"), "horse": ("foal", "foals"), "kangaroo": ("joey", "joeys"),
    "swan": ("cygnet", "cygnets"), "owl": ("owlet", "owlets"), "chicken": ("chick", "chicks"),
    "goat": ("kid", "kids"), "eagle": ("eaglet", "eaglets"), "butterfly": ("caterpillar", "caterpillars"),
}
NOUN_PLURAL = {
    "child": "children", "mouse": "mice", "man": "men", "woman": "women", "foot": "feet",
    "tooth": "teeth", "person": "people", "box": "boxes", "city": "cities", "knife": "knives",
    "leaf": "leaves", "wolf": "wolves", "baby": "babies", "bus": "buses", "potato": "potatoes",
    "puppy": "puppies", "kitten": "kittens", "calf": "calves", "lamb": "lambs", "piglet": "piglets",
    "duckling": "ducklings", "tadpole": "tadpoles", "cub": "cubs", "fawn": "fawns",
    "gosling": "goslings", "foal": "foals", "joey": "joeys", "cygnet": "cygnets", "owlet": "owlets",
    "chick": "chicks", "kid": "kids", "eaglet": "eaglets", "caterpillar": "caterpillars",
    "goose": "geese", "life": "lives", "hero": "heroes", "story": "stories", "church": "churches",
}

OBJECT_COLOR = {
    "banana": "yellow", "grass": "green", "snow": "white", "coal": "black", "blood": "red",
    "the sky": "blue", "a lemon": "yellow", "milk": "white", "a tomato": "red", "a carrot": "orange",
    "a crow": "black", "a strawberry": "red", "a lime": "green", "an emerald": "green",
    "a sapphire": "blue", "chocolate": "brown", "a pumpkin": "orange", "an eggplant": "purple",
}
COLOR_FR = {"yellow": "jaune", "green": "vert", "white": "blanc", "black": "noir", "red": "rouge",
            "blue": "bleu", "orange": "orange", "brown": "marron", "purple": "violet"}

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAY_FR = {"Monday": "lundi", "Tuesday": "mardi", "Wednesday": "mercredi", "Thursday": "jeudi",
          "Friday": "vendredi", "Saturday": "samedi", "Sunday": "dimanche"}

NUM_WORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight",
            9: "nine", 10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
            15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen",
            20: "twenty", 21: "twenty-one", 22: "twenty-two", 24: "twenty-four", 26: "twenty-six",
            28: "twenty-eight", 30: "thirty"}
NUM_FR = {1: "un", 2: "deux", 3: "trois", 4: "quatre", 5: "cinq", 6: "six", 7: "sept", 8: "huit",
          9: "neuf", 10: "dix", 11: "onze", 12: "douze", 13: "treize", 14: "quatorze",
          15: "quinze", 16: "seize", 17: "dix-sept", 18: "dix-huit", 19: "dix-neuf", 20: "vingt",
          22: "vingt-deux", 24: "vingt-quatre", 26: "vingt-six", 28: "vingt-huit", 30: "trente"}

CAPITAL_TO_COUNTRY = {v[0]: k for k, v in COUNTRIES.items()}
_EN_TO_FR = {**{a: v[2] for a, v in ADJ.items()}, **COLOR_FR, **DAY_FR}

# ----------------------------------------------------------------------------- skills
# Each skill: in_type, out_type, function (None if undefined), NP template, cue phrases.
# Types: country, city, language, continent, currency, adj, verb, animal, noun, object,
# color, day, num, word_en (any English word), word_fr, letter.
STR_TYPES = {"country", "city", "language", "continent", "currency", "adj", "verb", "noun",
             "color", "day", "word_en", "word_fr", "animal"}


def _first(s):
    return s[0].upper()


def _last(s):
    return s[-1].upper()


SKILLS = {
    "capital": dict(inp={"country"}, out="city", f=lambda x: COUNTRIES[x][0],
                    np="the capital of {x}", cues=["capital of", "capital city"]),
    "country_of_capital": dict(inp={"city"}, out="country", f=lambda x: CAPITAL_TO_COUNTRY.get(x),
                               np="the country whose capital is {x}", cues=["is the capital of"]),
    "language": dict(inp={"country"}, out="language", f=lambda x: COUNTRIES[x][1],
                     np="the main language spoken in {x}", cues=["language spoken", "official language"]),
    "continent": dict(inp={"country"}, out="continent", f=lambda x: COUNTRIES[x][2],
                      np="the continent that {x} is in", cues=["continent"]),
    "currency": dict(inp={"country"}, out="currency", f=lambda x: COUNTRIES[x][3],
                     np="the currency of {x}", cues=["currency of", "currency is"]),
    "opposite": dict(inp={"adj", "verb"}, out="same",
                     f=lambda x: ADJ[x][0] if x in ADJ else (VERB[x][0] if x in VERB else None),
                     np="the opposite of {x}", cues=["opposite of", "antonym"]),
    "comparative": dict(inp={"adj"}, out="adj_cmp", f=lambda x: ADJ.get(x, (None, None))[1],
                        np="the comparative form of {x}", cues=["comparative"]),
    "past_tense": dict(inp={"verb"}, out="verb_past", f=lambda x: VERB.get(x, (None, None))[1],
                       np="the past tense of {x}", cues=["past tense"]),
    "plural": dict(inp={"noun"}, out="noun_pl", f=lambda x: NOUN_PLURAL.get(x),
                   np="the plural of {x}", cues=["plural of", "plural form"]),
    "baby_animal": dict(inp={"animal"}, out="noun", f=lambda x: ANIMAL[x][0] if x in ANIMAL else None,
                        np="the word for a baby {x}", cues=["baby animal", "young animal"]),
    "to_french": dict(inp={"adj", "color", "day", "num"}, out="word_fr",
                      f=lambda x: (NUM_FR.get(x) if isinstance(x, int) else _EN_TO_FR.get(x)),
                      np="the French word for {x}", cues=["in French", "French word"]),
    "first_letter": dict(inp=STR_TYPES | {"num_word"}, out="letter", f=_first,
                         np="the first letter of {x}", cues=["first letter", "starts with the letter"]),
    "last_letter": dict(inp=STR_TYPES | {"num_word"}, out="letter", f=_last,
                        np="the last letter of {x}", cues=["last letter", "ends with the letter"]),
    "successor": dict(inp={"num"}, out="num", f=lambda x: x + 1,
                      np="the number after {x}", cues=["the number after", "plus one"]),
    "double": dict(inp={"num"}, out="num", f=lambda x: 2 * x,
                   np="two times {x}", cues=["two times", "twice as"]),
    "number_word": dict(inp={"num"}, out="num_word", f=lambda x: NUM_WORD.get(x),
                        np="the English word for {x}", cues=["in words", "spelled out"]),
    "next_day": dict(inp={"day"}, out="day", f=lambda x: DAYS[(DAYS.index(x) + 1) % 7],
                     np="the day after {x}", cues=["the day after"]),
    "color_of": dict(inp={"object"}, out="color", f=lambda x: OBJECT_COLOR.get(x),
                     np="the color of {x}", cues=["color of", "colour of"]),
}

# Base domains (inputs) per input type
DOMAIN = {
    "country": list(COUNTRIES), "city": list(CAPITAL_TO_COUNTRY), "adj": list(ADJ),
    "verb": list(VERB), "animal": list(ANIMAL), "noun": list(NOUN_PLURAL), "object": list(OBJECT_COLOR),
    "day": DAYS, "num": list(range(1, 13)),
}

# Ordered compositions outer(inner(x)) on a chosen base domain. One composition per
# unordered skill pair. Pre-registered list (35 pairs).
PAIRS = [
    # outer, inner, base domain of x
    ("first_letter", "capital", "country"),
    ("first_letter", "country_of_capital", "city"),
    ("first_letter", "language", "country"),
    ("first_letter", "currency", "country"),
    ("first_letter", "opposite", "adj"),
    ("first_letter", "to_french", "adj"),
    ("first_letter", "number_word", "num"),
    ("first_letter", "next_day", "day"),
    ("first_letter", "baby_animal", "animal"),
    ("first_letter", "color_of", "object"),
    ("first_letter", "continent", "country"),
    ("last_letter", "capital", "country"),
    ("last_letter", "country_of_capital", "city"),
    ("last_letter", "language", "country"),
    ("last_letter", "currency", "country"),
    ("last_letter", "opposite", "adj"),
    ("last_letter", "to_french", "adj"),
    ("last_letter", "number_word", "num"),
    ("last_letter", "next_day", "day"),
    ("last_letter", "baby_animal", "animal"),
    ("last_letter", "color_of", "object"),
    ("language", "country_of_capital", "city"),
    ("continent", "country_of_capital", "city"),
    ("currency", "country_of_capital", "city"),
    ("comparative", "opposite", "adj"),
    ("to_french", "opposite", "adj"),
    ("past_tense", "opposite", "verb"),
    ("plural", "baby_animal", "animal"),
    ("to_french", "color_of", "object"),
    ("to_french", "next_day", "day"),
    ("to_french", "successor", "num"),
    ("to_french", "double", "num"),
    ("number_word", "successor", "num"),
    ("number_word", "double", "num"),
    ("successor", "double", "num"),
]
