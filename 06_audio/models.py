import sys
import time

import torch

#---Device check---
def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"



#---Adaptaters---

class WhisperAdaptater:
    pass


class QwenASRAdaptater:
    pass



#---Model registery---
MODELS = {

}

def load_model(name, device = None):
    pass



#---Dunder secu and tests---

def main():
    pass


if __name__ == '__main__':
    main()