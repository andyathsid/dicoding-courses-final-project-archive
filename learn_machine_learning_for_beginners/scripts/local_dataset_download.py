import kagglehub
from pathlib import Path
import shutil

def download_dataset(project_root_path: str | Path | None = None):
    """Download dataset dari kaggle dan simpan di folder data
    
    Args:
        project_root_path: Path opsional dari project root. Jika None, menggunakan parent directory dari folder script ini.
    """
    if project_root_path is None:
        project_root = Path(__file__).resolve().parent.parent
    else:
        project_root = Path(project_root_path).resolve()
    
    data_dir = project_root / "data"
    downloaded_path = Path(kagglehub.dataset_download("mashlyn/online-retail-ii-uci"))
    
    data_dir.mkdir(parents=True, exist_ok=True)

    for src in downloaded_path.glob('*'):
        if src.is_file():
            dest = data_dir / src.name
            
            if not dest.exists() or (src.stat().st_size != dest.stat().st_size):
                shutil.copy(src, dest)
                print(f"{src.name} di-copy ke {dest}")
            else:
                print(f"{src.name} sudah ada di {dest}")
            
    print(f"Lokasi cache: {downloaded_path}")
    
if __name__ == "__main__":
    download_dataset()