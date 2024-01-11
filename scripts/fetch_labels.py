from pathlib import Path
from src.pressure_damping.label_processer import FirebaseCurveFetcher


if __name__ == '__main__':
    fetcher = FirebaseCurveFetcher(Path('.firebase.json'))

    curves = fetcher.fetch('imp-coro-flow-inv')
    curve = curves[0] 

    