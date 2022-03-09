import os
from functools import wraps
import urllib.request


def download_asset(
    filename,
    folder_prefix=".",
    prefix="test_assets/",
    bucket_url="https://locobot-bucket.s3.amazonaws.com/",
):
    prefix_url = bucket_url + prefix
    final_url = prefix_url + filename
    final_path = os.path.join(folder_prefix, filename)
    os.makedirs(folder_prefix, exist_ok=True)
    urllib.request.urlretrieve(final_url, final_path)


_downloaded = set()


def download_assets(assets, folder_prefix, url_prefix, refresh=False):
    global _downloaded

    status = False
    failed = False
    try:
        for asset in assets:
            full_path = os.path.join(folder_prefix, asset)
            if full_path not in _downloaded:
                if refresh or not os.path.isfile(full_path):
                    _downloaded.add(full_path)
                    download_asset(
                        asset,
                        folder_prefix=folder_prefix,
                        prefix=url_prefix,
                    )

    except:
        failed = True
    return not failed


def skipIfOfflineDecorator(files, folder_prefix, url_prefix):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if not download_assets(files, folder_prefix, url_prefix):
                raise unittest.SkipTest("Not able to download test assets, so skipping test")
            else:
                return fn(*args, **kwargs)

        return wrapper

    return decorator
