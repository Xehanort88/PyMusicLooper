from typing import Optional


class LoopNotFoundError(Exception):
    def __init__(self, message, reason: Optional[str] = None, detail: Optional[str] = None):
        """
        Args:
            message: The full error message.
            reason (str, optional): A short reason shared by the files failing the same way (e.g. "missing loop tags"),
            used to group them in the batch summary.
            detail (str, optional): Details specific to the file, shown next to it in the batch summary.
        """
        super().__init__(message)
        self.reason = reason
        self.detail = detail


class AudioLoadError(Exception):
    def __init__(self, message):
        super().__init__(message)


class NotAudioError(AudioLoadError):
    """Raised when a file cannot be decoded as audio at all (e.g. a text or image file)."""
