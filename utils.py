"""
Utilities Module

Contains common utility functions used across the audio-to-text application.
"""

import os
import re
import yaml

from typing import Dict, Any, Optional
from urllib.parse import urlparse, parse_qs


def load_config(config_file: str = "config.yml") -> Dict[str, Any]:
    """
    Load and parse configuration from YAML file.

    Args:
        config_file (str): Path to the configuration file

    Returns:
        Dict[str, Any]: Parsed configuration data

    Raises:
        FileNotFoundError: If the configuration file does not exist
        yaml.YAMLError: If the configuration file is invalid
    """
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file '{config_file}' not found")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing configuration file: {str(e)}")


def ensure_dir(directory: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.

    Args:
        directory (str): Path to the directory to create
    """
    os.makedirs(directory, exist_ok=True)


def sanitize_filename(filename: str) -> str:
    """
    Remove invalid characters from filenames for safe file system operations.

    Args:
        filename (str): Original filename potentially containing unsafe characters

    Returns:
        str: Sanitized filename with invalid characters replaced by underscores
    """
    return re.sub(r'[\\/*?:"<>|]', "_", filename)


def get_youtube_video_id(url: str) -> Optional[str]:
    """
    Parse a YouTube URL to extract the unique video identifier.

    Handles various YouTube URL formats including youtu.be short links
    and standard youtube.com URLs with query parameters.

    Args:
        url (str): YouTube video URL in any standard format

    Returns:
        str: YouTube video ID or None if the URL format is unrecognized
    """
    parsed_url = urlparse(url)

    if parsed_url.netloc in ("youtu.be", "www.youtu.be"):
        return parsed_url.path[1:]

    if parsed_url.netloc in ("youtube.com", "www.youtube.com"):
        query_params = parse_qs(parsed_url.query)
        return query_params.get("v", [None])[0]

    return None
