import os

from PIL import Image


def save_to_file(
    given_image: Image.Image,
    given_directory: str,
    given_file_name: str,
) -> str:
    if not os.path.isdir(given_directory):
        os.mkdir(given_directory)
    derived_file_path = os.path.join(given_directory, given_file_name)
    given_image.save(derived_file_path)
    return derived_file_path
