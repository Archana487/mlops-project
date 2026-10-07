import argparse, json
from .train import rollback

def main():
    ap = argparse.ArgumentParser("prp")
    ap.add_argument("cmd", choices=["rollback"])
    a = ap.parse_args()
    if a.cmd == "rollback":
        print(json.dumps({"rolled_back_to": rollback()}))

if __name__ == "__main__":
    main()
