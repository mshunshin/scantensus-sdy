import firebase_admin
import firebase_admin.db

from pathlib import Path

from src.pressure_damping.types import *
from src.utils.matt_json import get_labels_from_firebase_project_data_new

@dataclass
class FetchResult:
    """
    The result of fetching curves.
    """

    config: Config
    """
    The config for the curves.
    """

    curves: list[Curve]
    """
    The curves.
    """

    matt_data: list[dict[str, any]]
    """
    The Matt-formated data.
    """

class FirebaseCurveFetcher:
    """
    Fetches curves from Firebase.
    """

    def __init__(self, credentials_path: Path) -> None:
        super().__init__()

        self.credentials_path = credentials_path
        self._setup_authentication()


    def _setup_authentication(self):
        """
        Sets up the Firebase authentication.
        """

        cred = firebase_admin.credentials.Certificate(self.credentials_path)

        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, { 'databaseURL': "https://scantensus.firebaseio.com" })


    def fetch(self, project_code: str) -> FetchResult:
        """
        Fetches the curves for the given project code.
        """

        config_data = firebase_admin.db.reference(f'/fiducial/{project_code}/config').get()
        project_data = firebase_admin.db.reference(f'/fiducial/{project_code}/labels').get()
        urls_data = firebase_admin.db.reference(f'/fiducial/{project_code}/urls').get()

        # maps (name -> url)
        url_map = {k: v['url'] for k, v in urls_data.items()}

        config = Config(
            access=config_data['access'],
            curves={k: CurveConfig(**v) for k, v in config_data['curves'].items()},
            name=config_data['name'],
            leaderboardSize=config_data['leaderboardSize'],
            imageStore=config_data['imageStore'],
            projectClass=config_data['projectClass'],
            accessEmailsColonInKeyButDotInValue=config_data['accessEmailsColonInKeyButDotInValue'],
        )

        #config = Config(**config_data)
        #config.curves = {k: CurveConfig(**v) for k, v in config.curves.items()}

        res = get_labels_from_firebase_project_data_new(project_data, project_code, {})

        output_curves: list[Curve] = []

        for curve in res:
            output_curves.append(Curve(
                project=project_code,
                file=curve['file'],
                user=curve['user'],
                time=curve['time'],
                label=curve['label'],
                type=curve['vis'],
                xs=[(float(value)) for value in curve['value_x'].split()],
                ys=[(float(value)) for value in curve['value_y'].split()],
                straight_flag=[(int(value)) for value in curve['straight_segment'].split()],
            ))

        return FetchResult(
            config=config,
            curves=output_curves,
            matt_data=res
        )

    def fetch_raw(self, project_code: str) -> dict[str, any]:
        """
        Fetches the raw data for the given project code.
        """

        return firebase_admin.db.reference(f'/fiducial/{project_code}/labels').get()