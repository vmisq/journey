import os
import requests
import duckdb
import io
import pyarrow as pa
import numpy as np
from time import sleep
import zipfile
from PIL import Image
import cv2
import pytesseract

API_KEY = os.environ.get('API_KEY')
ROOT_FOLDER_ID = os.environ.get('ROOT_FOLDER_ID')
NAME_1 = os.environ.get('NAME_1')
HEIGHT_1 = os.environ.get('HEIGHT_1')
HEIGHT_OTHER = os.environ.get('HEIGHT_OTHER')

CODE_VERSION = {
    os.environ.get('FOLDER_1'): '0.1',
    os.environ.get('FOLDER_2'): '0.1',
    os.environ.get('FOLDER_3'): '0.1',
    os.environ.get('FOLDER_4'): '0.1',
    os.environ.get('FOLDER_5'): '0.1'
}

def list_to_duck(pylist):
    return duckdb.arrow(pa.Table.from_pylist(pylist))

def duck_to_list(duck):
    return duck.arrow().to_pylist()

def list_files(folder_id):
    url = "https://www.googleapis.com/drive/v3/files"
    params = {
        "q": f"'{folder_id}' in parents",
        "fields": "files(id, name, kind, mimeType, createdTime, modifiedTime)",
        "key": API_KEY
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    f = r.json().get("files", [])
    return r.json().get("files", [])

def filter_out_processed_files(all_files, folder, v='0'):
    t_all = list_to_duck(all_files)
    try:
        t_files = duckdb.sql(f"""
            SELECT *
            FROM t_all
            WHERE id NOT IN (
                SELECT id
                FROM p.log_parsed_items
                WHERE folder = '{folder}'
                AND _version = '{v}'
            )
        """)
        return duck_to_list(t_files)
    except duckdb.Error as e:
        if 'does not exist' in str(e):
            return duck_to_list(t_all)
        else:
            raise e

def list_subfolders():
    r = list_files(ROOT_FOLDER_ID)
    return {i['name']: i['id'] for i in r}

def write_log_parsed_items(files, folder, v='0'):
    t = list_to_duck(files)
    duckdb.sql(f"""
        CREATE TABLE IF NOT EXISTS p.log_parsed_items AS
        SELECT '{folder}' AS "folder", "kind", "id", "name", "mimeType", "bytes", "createdTime", "modifiedTime", CURRENT_TIMESTAMP AS "_timestamp", '{v}' AS "_version"
        FROM t
        LIMIT 1
        WITH NO DATA
    """)
    duckdb.sql(f"""
        INSERT INTO p.log_parsed_items ("folder", "kind", "id", "name", "mimeType", "bytes", "createdTime", "modifiedTime", "_timestamp", "_version")
        SELECT '{folder}' AS "folder", "kind", "id", "name", "mimeType", "bytes", "createdTime", "modifiedTime", CURRENT_TIMESTAMP AS "_timestamp", '{v}' AS "_version"
        FROM t
    """)

def download_file(file_id):
    download_url = f"https://drive.google.com/uc?id={file_id}"
    r = requests.get(download_url)
    r.raise_for_status()
    return r.content

def image_preprocessing(image, custom_config=None):
    img = np.array(image.convert("L"))
    img = cv2.convertScaleAbs(img, alpha=2, beta=0)
    _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = Image.fromarray(img)
    
    text = pytesseract.image_to_string(img, config=custom_config).replace(' ', '').replace('\x0c', '')
    return text, img

def parse_scale_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))

        header_box = (200, 100, 1000, 300)
        header_text, header_img = image_preprocessing(
            image.crop(header_box),
            custom_config=r'--oem 3 --psm 6'
        )
        name = header_text.split('\n')[0]
        date = header_text.split('\n')[1]
        if name == NAME_1:
            date = date[6:10] + '/' + date[:2] + '/' + date[3:5] + ' ' + date[10:]
            weight_box = (300, 300, 800, 600)
            imc_bf_box = (0, 1200, 1080, 1400)
            breakdown_box = (500, 1700, 1080, 2500)
        else:
            date = date[6:10] + '/' + date[3:5] + '/' + date[:2] + ' ' + date[10:]
            weight_box = (300, 300, 800, 600)
            imc_bf_box = (0, 1100, 1080, 1400)
            breakdown_box = (500, 1400, 1080, 2200)

        weight_text, weight_img = image_preprocessing(
            image.crop(weight_box),
            custom_config=r'--oem 3 --psm 11 -c tessedit_char_whitelist=0123456789'
        )
        weight = int(weight_text.split('\n')[0])
        weight = weight / 10 if weight > 99 else weight

        imc_bf_text, imc_bf_img = image_preprocessing(
            image.crop(imc_bf_box),
            custom_config=r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789,>'
        )
        imc_text = imc_bf_text.split('>')[0].replace(',', '.')[:4]
        bf = float(imc_bf_text.split('>')[1].replace(',', '.')[:4])
        if imc_text != '':
            imc = float(imc_text)
        elif name == NAME_1:
            imc = weight / (HEIGHT_1 * HEIGHT_1)
        else:
            imc = weight / (HEIGHT_OTHER * HEIGHT_OTHER)
        imc = imc / 10 if imc > 99 else imc
        bf = bf / 10 if bf > 99 else bf
        
        breakdown_text, breakdown_img = image_preprocessing(
            image.crop(breakdown_box),
            custom_config=r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789,'
        )
        if len([text for text in breakdown_text.split('\n') if ',' in text]) == 4:
            water, fat, bone, protein = [float(text.replace(',', '.')) for text in breakdown_text.split('\n') if ',' in text]
        elif len([text for text in breakdown_text.split('\n') if text not in ['', '0', '1', '5', '6', '\x0c']]) == 4:
            water, fat, bone, protein = [float(text.replace(',', '.')) for text in breakdown_text.split('\n') if text not in ['', '0', '1', '5', '6', '\x0c']]
        else:
            breakdown = []
            for bd in breakdown_text.split('\n'):
                if ',' in bd:
                    breakdown.append(float(bd.replace(',', '.')))
                elif int(bd) if str(bd).isdigit() else 0 > 10:
                    breakdown.append(int(bd) / 10)
            water, fat, bone, protein = breakdown[:4]
        water = water / 10 if water > 99 else water
        fat = fat / 10 if fat > 99 else fat
        bone = bone / 10 if bone > 9 else bone
        protein = protein / 10 if protein > 29 else protein

        yield ('', list_to_duck([{
            'name': name,
            'date': date,
            'weight': round(weight, 1),
            'imc': round(imc, 1),
            'bf': round(bf, 1),
            'water': round(water, 1),
            'fat': round(fat, 1),
            'bone': round(bone, 1),
            'protein': round(protein, 1),
        }]))
    except Exception as e:
        print(e)
        display(image)
        display(header_img)
        print(header_text)
        print({'res': [name, date]})
        display(weight_img)
        print(weight_text)
        print({'res': [weight]})
        display(imc_bf_img)
        print(imc_bf_text)
        print({'res': [imc, bf]})
        display(breakdown_img)
        print(breakdown_text)
        print({'res': [water, fat, bone, protein]})

def parse_loseit_files(zip_bytes):
    zip_bytes_io = io.BytesIO(zip_bytes)
    with zipfile.ZipFile(zip_bytes_io) as zf:
        for file_name in ['weights', 'food-logs', 'exercise-logs']:
            with zf.open(file_name+'.csv') as f:
                f_table = duckdb.read_csv(io.StringIO(f.read().decode('utf-8')))
                yield (file_name.replace('-', '_'), f_table)

def parse_strong_files(csv_bytes):
    f_table = duckdb.read_csv(io.StringIO(csv_bytes.decode('utf-8')))
    f_out = duckdb.sql("""
        SELECT *, row_number() OVER () _row_number FROM f_table
    """) 
    yield ('', f_out)

def process_files(subfolder, folder, target, processor):
    v = CODE_VERSION[folder]
    all_files = list_files(subfolder[folder])
    if not all_files:
        print(f'No items in {folder}')
        return None
    file_list = filter_out_processed_files(all_files, folder, v=v)
    data_list = []
    targets = set()
    processed_list = []
    for file in file_list:
        try:
            file_id = file['id']
            file_content = download_file(file_id)
            file_data = processor(file_content)
            file_byte = ''.join(format(byte, '08b') for byte in file_content)
            for f_data in file_data:
                _target = f_data[0]
                data_list.append({'id': file_id, '_target': _target, 'duck': f_data[1]})
                targets.add(_target)
            processed_list.append(file | {'bytes': file_byte})
            print(file_id)
        except Exception as e:
            print(file_id)
            print(e)
    for suffix in targets:
        combined_data = []
        for item in data_list:
            if item['_target'] == suffix:
                partial_data = item['duck']
                partial_data_md = duckdb.sql(f"""
                    SELECT '{item['id']}' AS id, '{folder}' AS folder, *, CURRENT_TIMESTAMP AS "_timestamp", '{v}' AS "_version"
                    FROM partial_data
                """)
                combined_data.append(partial_data_md)
        t_data = combined_data[0]
        for cd in combined_data[1:]:
            t_data = t_data.union(cd)
        tgt = f"{target}{'_' if suffix != '' else ''}{suffix}"
        cols = ','.join([f'"{c}"' for c in t_data.columns if c != '_target'])
        n = 0
        while n < 5:
            try:
                duckdb.sql(f"""
                    CREATE TABLE IF NOT EXISTS p.{tgt} AS
                    SELECT {cols}
                    FROM t_data
                    LIMIT 1
                    WITH NO DATA
                """)
                duckdb.sql(f"""
                    INSERT INTO p.{tgt} ({cols})
                    SELECT {cols}
                    FROM t_data
                """)
                break
            except Exception as e:
                print(e)
                print(cols)
                print(tgt)
                n += 1
                if n == 5:
                    raise e
        write_log_parsed_items(processed_list, folder, v=v)
    else:
        print(f'No new items in {folder}')

from prefect import flow, task
from prefect.futures import wait

@task
def get_subfolders():
    return list_subfolders()

@task
def process_each(subfolders, folder, target, function):
    process_files(subfolders, folder, target, function)

@flow
def drive_to_pg():
    try:
        duckdb.sql("DETACH p")
    except duckdb.BinderException:
        pass
    duckdb.sql("ATTACH '' AS p (TYPE postgres)")
    subfolders = get_subfolders()
    task_list = [
        process_each.submit(subfolder=subfolders, folder=os.environ.get('FOLDER_1'), target=os.environ.get('TARGET_1'), function=parse_scale_image),
        process_each.submit(subfolder=subfolders, folder=os.environ.get('FOLDER_2'), target=os.environ.get('TARGET_2'), function=parse_loseit_files),
        process_each.submit(subfolder=subfolders, folder=os.environ.get('FOLDER_3'), target=os.environ.get('TARGET_3'), function=parse_loseit_files),
        process_each.submit(subfolder=subfolders, folder=os.environ.get('FOLDER_4'), target=os.environ.get('TARGET_4'), function=parse_strong_files),
        process_each.submit(subfolder=subfolders, folder=os.environ.get('FOLDER_5'), target=os.environ.get('TARGET_5'), function=parse_strong_files),
    ]
    wait(task_list)
