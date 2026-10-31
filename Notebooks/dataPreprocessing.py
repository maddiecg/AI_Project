"""Clean the Fashion Product Images (Small) dataset for Designer AI.

Run from anywhere:
    python Backend/data_processing.py

Requirements: pandas, Pillow
The script discovers styles.csv / styles.zip and images/ / images.zip
inside the project's Data directory, then writes Data/processed/clothing_data.csv.
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name.lower() == 'backend' else Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / 'Data'

# Original dataset uses masterCategory and subCategory / articleType.
# Prioritize articleType (specific), falling back to subCategory.
TOP_TYPES = {'tshirts', 'shirts', 'tops', 'sweatshirts', 'sweaters', 'kurtas', 'tunics', 'polos', 'tank tops', 'blouses', 'hoodies'}
BOTTOM_TYPES = {'jeans', 'trousers', 'shorts', 'skirts', 'track pants', 'leggings', 'capris', 'jeggings', 'salwar', 'churidar'}
JACKET_TYPES = {'jackets', 'blazers', 'coats', 'waistcoat', 'waistcoats', 'rain jackets', 'suits'}
DRESS_TYPES = {'dresses', 'dress', 'jumpsuit', 'jumpsuits'}


def normalized(value: object) -> str:
    if pd.isna(value):
        return ''
    return ' '.join(str(value).strip().lower().replace('-', ' ').split())


def classify(row: pd.Series) -> str | None:
    article = normalized(row.get('articleType'))
    sub = normalized(row.get('subCategory'))
    for label, types in [('top', TOP_TYPES), ('bottom', BOTTOM_TYPES), ('jacket', JACKET_TYPES), ('dress', DRESS_TYPES)]:
        if article in types:
            return label
    # Fall back to broad categories only when the article type is missing.
    if not article:
        if sub in {'topwear'}:
            return 'top'
        if sub in {'bottomwear'}:
            return 'bottom'
        if sub in {'dress', 'dresses'}:
            return 'dress'
    return None


def discover_csv(data_dir: Path) -> tuple[str, Path, str | None]:
    candidates = sorted(data_dir.rglob('*.csv'))
    candidates = [p for p in candidates if p.name.lower() == 'styles.csv' and 'processed' not in p.parts]
    if candidates:
        return 'file', candidates[0], None
    for archive in sorted(data_dir.rglob('*.zip')):
        if 'processed' in archive.parts:
            continue
        with zipfile.ZipFile(archive) as zf:
            matches = [name for name in zf.namelist() if Path(name).name.lower() == 'styles.csv']
            if matches:
                return 'zip', archive, matches[0]
    raise FileNotFoundError('Cannot find styles.csv or a ZIP containing styles.csv inside ' + str(data_dir))


def load_styles(data_dir: Path) -> pd.DataFrame:
    kind, source, member = discover_csv(data_dir)
    if kind == 'file':
        return pd.read_csv(source, on_bad_lines='skip', low_memory=False, encoding='utf-8-sig')
    with zipfile.ZipFile(source) as zf:
        with zf.open(member) as f:
            return pd.read_csv(f, on_bad_lines='skip', low_memory=False, encoding='utf-8-sig')


def find_images(data_dir: Path) -> dict[str, tuple[str, Path, str | None]]:
    """Map normalized image filename to ('file'/'zip', source path, member)."""
    images = {}
    for path in sorted(data_dir.rglob('*')):
        if 'processed' in path.parts:
            continue
        if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
            images.setdefault(path.name.lower(), ('file', path, None))
    for archive in sorted(data_dir.rglob('*.zip')):
        if 'processed' in archive.parts:
            continue
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                name = Path(member).name.lower()
                if Path(name).suffix in {'.jpg', '.jpeg', '.png'}:
                    images.setdefault(name, ('zip', archive, member))
    return images


def check_image(location: tuple[str, Path, str | None]) -> bool:
    kind, source, member = location
    try:
        if kind == 'file':
            with Image.open(source) as im:
                im.verify()
        else:
            with zipfile.ZipFile(source) as zf:
                with zf.open(member) as f:
                    with Image.open(io.BytesIO(f.read())) as im:
                        im.verify()
        return True
    except (OSError, ValueError, UnidentifiedImageError, zipfile.BadZipFile):
        return False


def process_data(data_dir: str | Path = DEFAULT_DATA_DIR, output: str | Path | None = None, verify_images: bool = False) -> pd.DataFrame:
    data_dir = Path(data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f'Data folder not found: {data_dir}')
    styles = load_styles(data_dir)
    required = {'id', 'articleType', 'subCategory'}
    missing = required - set(styles.columns)
    if missing:
        raise ValueError(f'styles.csv is missing expected columns: {sorted(missing)}')

    styles = styles.copy()
    styles['id'] = pd.to_numeric(styles['id'], errors='coerce')
    styles = styles.dropna(subset=['id'])
    styles = styles[styles['id'].mod(1).eq(0)].copy()
    styles['id'] = styles['id'].astype('int64')
    styles = styles.drop_duplicates(subset=['id'])
    styles['category'] = styles.apply(classify, axis=1)
    styles = styles.dropna(subset=['category']).copy()

    image_index = find_images(data_dir)
    rows = []
    for _, record in styles.iterrows():
        item_id = int(record['id'])
        location = next((image_index[f'{item_id}{ext}'] for ext in ('.jpg', '.jpeg', '.png') if f'{item_id}{ext}' in image_index), None)
        if location is None or (verify_images and not check_image(location)):
            continue
        kind, source, member = location
        # Archive paths are recorded as zip_path::internal/path, readable by downstream code.
        image_path = str(source) if kind == 'file' else f'{source}::{member}'
        rows.append({
            'id': item_id,
            'image_path': image_path,
            'category': record['category'],
            'article_type': str(record.get('articleType', '') or ''),
            'sub_category': str(record.get('subCategory', '') or ''),
            'base_colour': str(record.get('baseColour', '') or ''),
            'season': str(record.get('season', '') or ''),
            'usage': str(record.get('usage', '') or ''),
            'product_display_name': str(record.get('productDisplayName', '') or ''),
        })

    columns = ['id', 'image_path', 'category', 'article_type', 'sub_category', 'base_colour', 'season', 'usage', 'product_display_name']
    cleaned = pd.DataFrame(rows, columns=columns).sort_values('id').reset_index(drop=True)
    if cleaned.empty:
        raise ValueError('No matching clothing images were found. Verify styles.csv and the images ZIP/folder are under Data/.')
    destination = Path(output).expanduser().resolve() if output else data_dir / 'processed' / 'clothing_data.csv'
    destination.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(destination, index=False)
    print(f'Saved {len(cleaned)} labeled clothing images to {destination}')
    print(cleaned['category'].value_counts().to_string())
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare Fashion Product Images for Designer AI.')
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--verify-images', action='store_true', help='Verify each image can be decoded (slower).')
    args = parser.parse_args()
    process_data(args.data_dir, args.output, args.verify_images)


if __name__ == '__main__':
    main()
