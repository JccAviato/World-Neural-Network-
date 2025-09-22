import argparse, random, csv

def load_countries(csv_path):
    with open(csv_path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def ask_country_from_capital(countries):
    c = random.choice(countries)
    question = f"Which country has capital '{c['capital']}'?"
    opts = [c['country']]
    pool = [x['country'] for x in countries if x['country'] != c['country']]
    opts += random.sample(pool, k=min(3, len(pool)))
    random.shuffle(opts)
    return question, opts, c['country']

def ask_continent(countries):
    c = random.choice(countries)
    question = f"'{c['country']}' belongs to which continent?"
    conts = sorted(list(set(x['continent'] for x in countries)))
    correct = c['continent']
    opts = [correct] + random.sample([x for x in conts if x != correct], k=min(3, len(conts)-1))
    random.shuffle(opts)
    return question, opts, correct

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--countries_csv", default="data/countries.csv")
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()
    countries = load_countries(args.countries_csv)
    score = 0
    for i in range(1, args.n+1):
        if random.random() < 0.5:
            q, opts, ans = ask_country_from_capital(countries)
        else:
            q, opts, ans = ask_continent(countries)
        print(f"Q{i}: {q}")
        for j, o in enumerate(opts):
            print(f"  {j+1}. {o}")
        try:
            pick = int(input("Your answer (1-4): ").strip())
            sel = opts[pick-1]
        except Exception:
            sel = None
        if sel == ans:
            print("✅ Correct!\n")
            score += 1
        else:
            print(f"❌ Incorrect. Answer: {ans}\n")
    print(f"Final score: {score}/{args.n}")

if __name__ == "__main__":
    main()
