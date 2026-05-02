"""
File handling utilities for the API
"""
from typing import Tuple
from fastapi import UploadFile
import magic


class FileHandler:
    """Handles file validation and processing"""

    # Supported file types
    SUPPORTED_MIME_TYPES = {
        'application/pdf',
        'image/jpeg',
        'image/jpg',
        'image/png',
        'image/gif',
        'image/webp',
        'text/plain',
        'application/msword',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'application/vnd.ms-excel',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    }

    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

    @staticmethod
    async def validate_file(file: UploadFile) -> Tuple[bool, str]:
        """
        Validate uploaded file

        Args:
            file: Uploaded file from FastAPI

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if file is provided
        if not file:
            return False, "No file provided"

        # Read file content
        content = await file.read()
        await file.seek(0)  # Reset file pointer

        # Check file size
        if len(content) > FileHandler.MAX_FILE_SIZE:
            return False, f"File size exceeds maximum allowed size of {FileHandler.MAX_FILE_SIZE / (1024*1024)}MB"

        # Check MIME type
        try:
            mime = magic.from_buffer(content, mime=True)
            if mime not in FileHandler.SUPPORTED_MIME_TYPES:
                return False, f"Unsupported file type: {mime}. Supported types: {', '.join(FileHandler.SUPPORTED_MIME_TYPES)}"
        except Exception as e:
            # If python-magic fails, fall back to content_type from upload
            if file.content_type not in FileHandler.SUPPORTED_MIME_TYPES:
                return False, f"Unsupported file type: {file.content_type}"

        return True, "File is valid"

    @staticmethod
    async def read_file(file: UploadFile) -> bytes:
        """
        Read file content

        Args:
            file: Uploaded file from FastAPI

        Returns:
            File content as bytes
        """
        content = await file.read()
        await file.seek(0)  # Reset file pointer
        return content
