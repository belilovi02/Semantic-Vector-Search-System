from pathlib import Path

def main():
    p = Path('experiments/results')
    files = sorted([f for f in p.glob('*H1_ingest*.json')])
    print('H1 JSON count:', len(files))
    for f in files:
        print(f.name)

if __name__ == '__main__':
    main()
