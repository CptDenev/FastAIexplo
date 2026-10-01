DEFAULT_MODEL = 'whisper-small'
DEFAULT_SESSION = './Records'


def main():

    model = DEFAULT_MODEL
    session = DEFAULT_SESSION
    
    while True :
        print("---Small model voice detection under radio like noise---")
        print("1. add noise to choosen session")
        print("2. choose transcription model")
        print("3. run model on choosen session")
        print("4. score all models on session (WER, keyword, latency)")
        choice = int(input("enter your choice :"))
        
        match choice:
            case 1:
                print('\n')
                print(f"noise added to file located at {session}")
                print('\n')

            case 2:
                print('\n')
                print(f"model choosen {model}")
                print('\n')

            case 3:
                print('\n')
                print(f"run {model} on session : {session}")
                print('\n')

            case 4:
                print('\n')
                print("launch benchmark on all models...")
                print('\n')
            
            case _:
                print('\n')
                print("exit program...")
                break



if __name__ == '__main__':
    main()