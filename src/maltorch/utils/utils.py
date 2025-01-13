import requests


def download_gdrive(gdrive_id, fname_save):
    """Extracted from RobustBench sourcecode https://github.com/RobustBench/robustbench.git."""

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value

        return None
    
    def get_uuid_token(response):
        # TODO: rude way, might want to implement proper HTML parsing
        if 'uuid" value="' not in response.text:
            return None
        
        uuid = response.text.split("uuid")[1].replace('" value="', "").split('"')[0]
        return uuid

    def save_response_content(response, fname_save):
        CHUNK_SIZE = 32768

        with open(fname_save, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    print("Download started: path={} (gdrive_id={})".format(fname_save, gdrive_id))

    docs_domain = "https://docs.google.com/uc"
    drive_domain = "https://drive.usercontent.google.com/download"
    session = requests.Session()

    response = session.get(docs_domain, params={"id": gdrive_id, "export": "download", "confirm": "t"}, stream=True)
    token = get_confirm_token(response)
    uuid = get_uuid_token(response)

    if token:
        params = {"id": gdrive_id, "confirm": token,  "export": "download"}
        response = session.get(docs_domain, params=params, stream=True)
    elif uuid:
        params = {"id": gdrive_id, "uuid": uuid, "export": "download", "confirm": "t"}
        response = session.get(drive_domain, params=params, stream=True)

    save_response_content(response, fname_save)
    session.close()
    print("Download finished: path={} (gdrive_id={})".format(fname_save, gdrive_id))
