import os

from io import (
    BytesIO,
)

from PIL import Image

from shortfin_apps.types.Base64CharacterEncodedByteSequence import (
    Base64CharacterEncodedByteSequence,
)


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


def png_from(given_image: Image.Image) -> Base64CharacterEncodedByteSequence:
    memory_for_png = BytesIO()
    given_image.save(memory_for_png, format="PNG")
    png_from_memory = memory_for_png.getvalue()
    return Base64CharacterEncodedByteSequence.decoded_from(png_from_memory)


def image_from(given_png: Base64CharacterEncodedByteSequence) -> Image.Image:
    memory_for_png = BytesIO(given_png.as_bytes)
    return Image.open(memory_for_png, formats=["PNG"])
