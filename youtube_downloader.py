"""
YouTube Downloader Module

Key features:
- Download audio from YouTube videos for transcription workflows
- Automatic sanitization of URLs to extract video IDs
- Support for both yt-dlp and wget download methods
- Customizable output directory organization
- Video information extraction (title, author, length, etc.)

Basic usage:
  # Download audio from a YouTube video (default to downloads/video_title.mp3)
  python youtube_downloader.py https://www.youtube.com/watch?v=VIDEO_ID

  # Show video information without downloading
  python youtube_downloader.py https://www.youtube.com/watch?v=VIDEO_ID --info

  # Specify custom output directory
  python youtube_downloader.py https://www.youtube.com/watch?v=VIDEO_ID --output-path custom_folder

  # Download full video with specific resolution (optional)
  python youtube_downloader.py https://www.youtube.com/watch?v=VIDEO_ID --resolution 720p

Requirements:
  - yt-dlp (required): pip install yt-dlp
  - wget (optional): brew install wget (macOS) or apt-get install wget (Linux)
"""

import os
import json
import subprocess
import shutil

from logger import Logger
from typing import Dict, Any
from utils import sanitize_filename, get_youtube_video_id, ensure_dir


class DownloadError(Exception):
    """Exception raised for errors during YouTube download operations."""

    pass


class YouTubeDownloader:
    """
    YouTube audio downloader optimized for transcription workflows.

    This class provides functionality to extract audio from YouTube videos
    using yt-dlp (with optional wget support), sanitize video URLs,
    and organize downloads in customizable directories.
    """

    def __init__(self, output_path: str = "downloads"):
        """
        Initialize the YouTubeDownloader with specified output directory.

        Args:
            output_path (str): Target directory where audio files will be saved.
                               Defaults to "downloads" in current directory.

        Raises:
            ImportError: If required dependencies are not installed
        """
        self.output_path = output_path
        ensure_dir(output_path)
        self._check_requirements()

    def _check_requirements(self) -> None:
        """
        Verify that required external tools are available on the system.

        Checks for yt-dlp (required) and wget (optional) availability,
        setting appropriate flags and displaying installation instructions
        if tools are missing.

        Raises:
            ImportError: If yt-dlp is not installed
        """
        self.has_wget = shutil.which("wget") is not None

        if not shutil.which("yt-dlp"):
            error_msg = "yt-dlp is not installed. This module requires yt-dlp."
            Logger.log(False, error_msg, "error")
            Logger.log(False, "Please install it using: pip install yt-dlp", "error")
            raise ImportError("yt-dlp is required but not installed")

    def get_video_info(self, url: str) -> Dict[str, Any]:
        """
        Retrieve comprehensive metadata about a YouTube video.

        Uses yt-dlp to fetch details including title, creator, duration,
        available quality options, view count, and thumbnail URLs.

        Args:
            url (str): YouTube video URL

        Returns:
            dict: Structured metadata including:
                - success: Boolean indicating operation success
                - title: Video title
                - author: Content creator name
                - length: Duration in seconds
                - thumbnail_url: URL to thumbnail image
                - available_resolutions: List of available quality options
                - views: View count
                - rating: Average rating
                Or error details if unsuccessful

        Raises:
            DownloadError: If video information cannot be retrieved
        """
        try:
            video_id = get_youtube_video_id(url)
            if not video_id:
                Logger.log(False, f"Invalid YouTube URL: {url}", "error")
                return {"success": False, "error": f"Invalid YouTube URL: {url}"}

            Logger.log(True, f"Getting info for video ID: {video_id}", "debug")

            cmd = ["yt-dlp", "--dump-json", url]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            info = json.loads(result.stdout)

            formats = info.get("formats", [])
            available_resolutions = []
            for fmt in formats:
                if fmt.get("height") and fmt.get("width"):
                    resolution = f"{fmt['height']}p"
                    if resolution not in available_resolutions:
                        available_resolutions.append(resolution)

            return {
                "success": True,
                "title": info.get("title", "Unknown"),
                "author": info.get("uploader", "Unknown"),
                "length": info.get("duration", 0),
                "thumbnail_url": info.get("thumbnail", ""),
                "available_resolutions": sorted(
                    available_resolutions, key=lambda x: int(x[:-1])
                ),
                "views": info.get("view_count", 0),
                "rating": info.get("average_rating", 0),
                "video_id": video_id,
            }
        except subprocess.CalledProcessError as e:
            error_msg = f"Error getting video info: {e.stderr}"
            Logger.log(False, error_msg, "error")
            return {"success": False, "error": error_msg}
        except json.JSONDecodeError as e:
            error_msg = f"Error parsing video info: {str(e)}"
            Logger.log(False, error_msg, "error")
            return {"success": False, "error": error_msg}
        except Exception as e:
            error_msg = f"Unexpected error retrieving video info: {str(e)}"
            Logger.log(False, error_msg, "error")
            return {"success": False, "error": error_msg}

    def download_video(self, url: str, resolution: str = "highest") -> Dict[str, Any]:
        """
        Download a complete YouTube video with audio and video tracks.

        Args:
            url (str): YouTube video URL
            resolution (str): Target quality setting, either "highest" for best available
                              or a specific resolution like "720p". Defaults to "highest".

        Returns:
            dict: Operation result containing:
                - success (bool): Whether download completed successfully
                - file_path (str): Path to downloaded file if successful
                - title (str): Video title if successful
                - error (str): Error details if unsuccessful

        Raises:
            DownloadError: If the download fails
        """
        try:
            Logger.log(True, "Getting video information")
            info = self.get_video_info(url)
            if not info["success"]:
                Logger.log(False, "Failed to get video information", "error")
                return info

            Logger.log(True, f"Video found: {info['title']}")
            Logger.log(True, "Video details", "debug")

            height = None
            if resolution != "highest" and resolution.endswith("p"):
                height = resolution[:-1]

            format_selector = "bestvideo+bestaudio/best"
            if height:
                if height in [res[:-1] for res in info["available_resolutions"]]:
                    format_selector = (
                        f"bestvideo[height<={height}]+bestaudio/best[height<={height}]"
                    )
                else:
                    Logger.log(
                        False,
                        f"Requested resolution {resolution} not available. Using best available.",
                        "warning",
                    )

            safe_title = sanitize_filename(info["title"])
            output_filename = f"{safe_title}.mp4"
            output_path = os.path.join(self.output_path, output_filename)

            Logger.log(True, f"Downloading video to: {output_path}")
            cmd = [
                "yt-dlp",
                "-f",
                format_selector,
                "-o",
                output_path,
                "--no-playlist",
                url,
            ]

            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            # Monitor download progress
            while True:
                output = process.stdout.readline()
                if output == "" and process.poll() is not None:
                    break
                # Skip diagnostic and technical messages from yt-dlp
                if output:
                    output_str = output.strip()
                    if (
                        not output_str.startswith("[ExtractAudio]")
                        and not output_str.startswith("[download]")
                        and "has already been downloaded" not in output_str
                    ):
                        print(f"\r{output_str}", end="")

            return_code = process.poll()
            if return_code != 0:
                stderr = process.stderr.read()
                error_msg = f"Download failed with code {return_code}: {stderr}"
                Logger.log(False, error_msg, "error")
                return {"success": False, "error": error_msg}

            Logger.log(True, "Download completed successfully")
            return {
                "success": True,
                "file_path": output_path,
                "title": info["title"],
            }

        except Exception as e:
            error_msg = f"Download error: {str(e)}"
            Logger.log(False, error_msg, "error")
            return {"success": False, "error": error_msg}

    def download_audio_only(self, url: str) -> Dict[str, Any]:
        """
        Download only the audio track from a YouTube video.

        This method is optimized for transcription workflows, extracting
        just the audio in a high-quality format suitable for speech recognition.

        Args:
            url (str): YouTube video URL

        Returns:
            dict: Operation result containing:
                - success (bool): Whether download completed successfully
                - file_path (str): Path to downloaded file if successful
                - title (str): Video title if successful
                - error (str): Error details if unsuccessful

        Raises:
            DownloadError: If the download fails
        """
        try:
            video_id = get_youtube_video_id(url)
            if not video_id:
                error_msg = f"Invalid YouTube URL: {url}"
                Logger.log(False, error_msg, "error")
                return {"success": False, "error": error_msg}

            Logger.log(True, f"Processing YouTube URL: {url}", "debug")
            Logger.log(True, f"Extracted video ID: {video_id}", "debug")

            info = self.get_video_info(url)
            if not info["success"]:
                return info

            title = info["title"]
            safe_title = sanitize_filename(title)
            output_filename = f"{safe_title}.mp3"
            output_path = os.path.join(self.output_path, output_filename)

            cmd = [
                "yt-dlp",
                "-f",
                "bestaudio",
                "-x",
                "--audio-format",
                "mp3",
                "--audio-quality",
                "0",
                "-o",
                output_path,
                "--no-playlist",
                "--quiet",
                url,
            ]

            Logger.log(True, f"Downloading audio from: {safe_title}")
            Logger.log(True, f"Output will be saved to: {output_path}")

            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            process.communicate()
            return_code = process.returncode

            if return_code != 0:
                stderr = (
                    process.stderr.read() if hasattr(process.stderr, "read") else ""
                )
                error_msg = f"Download failed with code {return_code}: {stderr}"
                Logger.log(False, error_msg, "error")
                return {"success": False, "error": error_msg}

            Logger.log(True, "Download completed successfully")
            return {
                "success": True,
                "file_path": output_path,
                "title": title,
            }

        except Exception as e:
            error_msg = f"Download error: {str(e)}"
            Logger.log(False, error_msg, "error")
            return {"success": False, "error": error_msg}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YouTube Downloader Tool")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument(
        "--info",
        action="store_true",
        help="Only show video information without downloading",
    )
    parser.add_argument(
        "--output-path",
        default="downloads",
        help="Output directory for downloaded files",
    )
    parser.add_argument(
        "--resolution",
        default="highest",
        help="Video resolution (e.g., 720p, 1080p) - only for full video download",
    )
    parser.add_argument(
        "--audio-only",
        action="store_true",
        help="Download audio only (optimized for transcription)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    if args.verbose:
        Logger.set_verbose(True)

    downloader = YouTubeDownloader(args.output_path)

    if args.info:
        video_info = downloader.get_video_info(args.url)
        if video_info["success"]:
            print("\nVideo Information:")
            print(f"Title: {video_info['title']}")
            print(f"Author: {video_info['author']}")
            print(f"Length: {video_info['length']} seconds")
            print(f"Views: {video_info['views']}")
            print(
                f"Available Resolutions: {', '.join(video_info['available_resolutions'])}"
            )
        else:
            print(f"\nError: {video_info['error']}")
    elif args.audio_only:
        result = downloader.download_audio_only(args.url)
        if result["success"]:
            print(f"\nAudio downloaded successfully to: {result['file_path']}")
        else:
            print(f"\nDownload failed: {result['error']}")
    else:
        result = downloader.download_video(args.url, args.resolution)
        if result["success"]:
            print(f"\nVideo downloaded successfully to: {result['file_path']}")
        else:
            print(f"\nDownload failed: {result['error']}")
