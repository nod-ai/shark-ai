import re
import binascii


class Base64CharacterEncodedByteSequence(str):
    """
    A sequence of 8-bit integers encoded using the [Base64 alphabet](https://www.rfc-editor.org/rfc/rfc4648.html#section-4).
    """

    pattern = (
        r"^(?:[A-Za-z0-9+\/]{4})*(?:[A-Za-z0-9+\/]{3}={1}|[A-Za-z0-9+\/]{2}={2})?$"
    )

    def __new__(Self, given_subject: str):
        if re.match(Self.pattern, given_subject) == None:
            raise ValueError("String cannot be interpreted as a byte sequence")

        return super().__new__(Self, given_subject)

    @property
    def as_bytes(self):
        base64_integer_encoded_byte_sequence = self.encode()

        derived_raw_byte_sequence = binascii.a2b_base64(
            base64_integer_encoded_byte_sequence
        )

        return derived_raw_byte_sequence

    @classmethod
    def decoded_from(Self, given_raw_byte_sequence: bytes):
        base64_integer_encoded_byte_sequence = binascii.b2a_base64(
            given_raw_byte_sequence, newline=False
        )

        base64_character_encoded_byte_sequence = (
            base64_integer_encoded_byte_sequence.decode()
        )

        return Self(base64_character_encoded_byte_sequence)
