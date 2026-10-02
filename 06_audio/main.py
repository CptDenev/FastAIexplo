from pathlib import Path

#from corpus import add_noise
#from models import MODELS
#from run import run_session
#from metrics import score_session

MODELS = ["whisper-base", "whisper-small", "qwen3-asr-0.6b"]
DEFAULT_MODEL = "whisper-small"
RECORDS_DIR = Path(__file__).resolve().parent / "Records"


#allow to pick a label from an option list and return it, choices starts at 1 for non dev people
def pick(options, label):
    for i, opt, in enumerate(options, 1):
        print(f"{i}. {opt}")

    try:
        idx = int(input(f"{label} : ")) -1
        if 0 <= idx < len(options):
            return options[idx]
    except ValueError:
        pass
    print("invalid choice")
    return None

#return all folders name in RECORDS_DIR
def list_sessions():
    return sorted(p.name for p in RECORDS_DIR.iterdir() if p.is_dir())


def main():

    model = DEFAULT_MODEL
    session = None
    
    while True :
        print("\n---Small model voice detection under radio like noise---")
        print(f"session : {session or 'none'} | model : {model}")
        print("1. choose a session")
        print("2. add noise to choosen session")
        print("3. choose transcription model")
        print("4. run model on choosen session")
        print("5. score all models on session (WER, keyword, latency)")
        print("0. quit")
        
        try:
            choice = int(input("enter your choice :"))
        except ValueError:
            print("please enter a number")
            continue


        if choice in (2,4,5) and session is None:
            print("choose a session first")
            continue

        
        match choice:
            case 1:
                session = pick(list_sessions(), "session") or session
            
            case 2:
                #add_noise(RECORDS_DIR / session)
                print('\n')
                print(f"noise added to {session}")
                print('\n')

            case 3:
                print('\n')
                model = pick(MODELS, "model") or model
                print(f"model choosen {model}")
                print('\n')

            case 4:
                #run_session(RECORDS_DIR / session, model)
                print('\n')
                print(f"run {model} on session : {session}")
                print('\n')

            case 5:
                #score_session(RECORDS_DIR / session)
                print('\n')
                print("launch benchmark on all models...")
                print('\n')
            
            case 0:
                print('\n')
                print("exit program...")
                break

            case _:
                print("invalid choice")



if __name__ == '__main__':
    main()