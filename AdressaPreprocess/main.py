from pathlib import Path
from tqdm import tqdm
import json

def main():
    data_path = Path("C:\\Users\Jiyun\Desktop\\NRS\datasets\\adressa\one_week")
    out_path = Path("")
    for file in data_path.iterdir():
        with open(file,"r") as f:
            for l in tqdm(f):
                event_dict = json.loads(l.strip("\n"))
                print(event_dict)
                break
    return


if __name__ == "__main__":
    main()
