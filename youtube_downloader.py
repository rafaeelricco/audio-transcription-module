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
import re
from logger import Logger

from urllib.parse import urlparse, parse_qs


class YouTubeDownloader:
    """
    YouTube audio downloader optimized for transcription workflows.

    This class provides functionality to extract audio from YouTube videos
    using yt-dlp (with optional wget support), sanitize video URLs,
    and organize downloads in customizable directories.
    """

    def __init__(self, output_path="downloads"):
        """
        Initialize the YouTubeDownloader with specified output directory.

        Args:
            output_path (str): Target directory where audio files will be saved.
                               Defaults to "downloads" in current directory.
        """
        self.output_path = output_path

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        self._check_requirements()

    def _check_requirements(self):
        """
        Verify that required external tools are available on the system.

        Checks for yt-dlp (required) and wget (optional) availability,
        setting appropriate flags and displaying installation instructions
        if tools are missing.
        """
        self.has_wget = shutil.which("wget") is not None
        # if not self.has_wget:
        #     print("Note: wget is not installed. Will use yt-dlp for downloads instead.")
        #     print(
        #         "To install wget (optional): brew install wget (macOS) or apt-get install wget (Ubuntu/Debian)"
        #     )

        if not shutil.which("yt-dlp"):
            print("Error: yt-dlp is not installed. This module requires yt-dlp.")
            print("Please install it using: pip install yt-dlp")
            raise ImportError("yt-dlp is required but not installed")

    def _sanitize_filename(self, filename):
        """
        Remove invalid characters from filenames for safe file system operations.

        Args:
            filename (str): Original filename potentially containing unsafe characters

        Returns:
            str: Sanitized filename with invalid characters replaced by underscores
        """
        return re.sub(r'[\\/*?:"<>|]', "_", filename)

    def _get_video_id(self, url):
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

    def get_video_info(self, url):
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
        """
        try:
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
            }
        except subprocess.CalledProcessError as e:
            return {"success": False, "error": f"Error getting video info: {e.stderr}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def download_video(self, url, resolution="highest"):
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
        """
        try:
            Logger.log(True, "Getting video information")
            info = self.get_video_info(url)
            if not info["success"]:
                Logger.log(False, "Failed to get video information")
                return info

            Logger.log(True, f"Video found: {info['title']}")
            Logger.log(True, "Video details")

            height = None
            if resolution != "highest" and resolution.endswith("p"):
                height = resolution[:-1]  # Remove the 'p' to get just the number

            format_selector = "bestvideo+bestaudio/best"
            if height:
                if height in [res[:-1] for res in info["available_resolutions"]]:
                    format_selector = (
                        f"bestvideo[height<={height}]+bestaudio/best[height<={height}]"
                    )
                else:
                    Logger.log(
                        False,
                        f"Resolution {resolution} not available. Using highest available resolution.",
                    )

            Logger.log(True, "Preparing download")
            cmd = ["yt-dlp", "-f", format_selector, "-g", url]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            direct_url = result.stdout.strip()

            safe_title = self._sanitize_filename(info["title"])
            output_file = os.path.join(self.output_path, f"{safe_title}.mp4")

            Logger.log(True, f"Downloading video: {resolution}")

            if self.has_wget:
                wget_cmd = [
                    "wget",
                    "-O",
                    output_file,
                    "--progress=bar:force",
                    direct_url,
                ]
                subprocess.run(wget_cmd, check=True)
            else:
                ytdlp_cmd = [
                    "yt-dlp",
                    "-f",
                    format_selector,
                    "-o",
                    output_file,
                    "--no-playlist",
                    url,
                ]
                subprocess.run(ytdlp_cmd, check=True)

            Logger.log(True, "Download complete")
            print(f"\n✓ Video downloaded successfully to: {output_file}")
            return {"success": True, "file_path": output_file, "title": info["title"]}

        except subprocess.CalledProcessError as e:
            Logger.log(False, "Download failed")
            error_msg = e.stderr if hasattr(e, "stderr") else str(e)
            print(f"\n✗ Download failed: {error_msg}")
            return {
                "success": False,
                "error": f"Error: {error_msg}",
                "file_path": None,
            }
        except Exception as e:
            Logger.log(False, "Download failed")
            print(f"\n✗ Download failed: {str(e)}")
            return {"success": False, "error": f"Error: {str(e)}", "file_path": None}

    def download_audio_only(self, url):
        """
        Extract and download only the audio track from a YouTube video.

        Optimized for transcription workflows, this method saves disk space
        by downloading only the audio component in MP3 format.

        Args:
            url (str): YouTube video URL

        Returns:
            dict: Operation result containing:
                - success (bool): Whether download completed successfully
                - file_path (str): Path to downloaded file if successful
                - title (str): Video title if successful
                - error (str): Error details if unsuccessful
        """
        try:
            Logger.log(True, "Getting video information")
            info = self.get_video_info(url)
            if not info["success"]:
                Logger.log(False, "Failed to get video information")
                return info

            Logger.log(True, f"Video found: {info['title']}")

            Logger.log(True, "Preparing audio extraction")
            cmd = ["yt-dlp", "-f", "bestaudio", "-g", url]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            direct_url = result.stdout.strip()

            safe_title = self._sanitize_filename(info["title"])
            output_file = os.path.join(self.output_path, f"{safe_title}.mp3")

            Logger.log(True, f"Downloading audio")

            if self.has_wget:
                wget_cmd = [
                    "wget",
                    "-O",
                    output_file,
                    "--progress=bar:force",
                    direct_url,
                ]
                subprocess.run(wget_cmd, check=True)
            else:
                ytdlp_cmd = [
                    "yt-dlp",
                    "-f",
                    "bestaudio",
                    "-o",
                    output_file,
                    "--no-playlist",
                    url,
                ]
                subprocess.run(ytdlp_cmd, check=True)

            Logger.log(True, "Download complete")
            print(f"\n✓ Audio downloaded successfully to: {output_file}")
            return {"success": True, "file_path": output_file, "title": info["title"]}

        except subprocess.CalledProcessError as e:
            Logger.log(False, "Download failed")
            error_msg = e.stderr if hasattr(e, "stderr") else str(e)
            print(f"\n✗ Audio download failed: {error_msg}")
            return {
                "success": False,
                "error": f"Error: {error_msg}",
                "file_path": None,
            }
        except Exception as e:
            Logger.log(False, "Download failed")
            print(f"\n✗ Audio download failed: {str(e)}")
            return {"success": False, "error": f"Error: {str(e)}", "file_path": None}

    def download(self, url):
        """
        Primary download method that extracts audio from YouTube videos.

        This is the recommended entry point for most use cases, as it
        focuses on audio extraction for transcription workflows.

        Args:
            url (str): YouTube video URL

        Returns:
            dict: Operation result with download status and file details
                  (See download_audio_only for returned dictionary structure)
        """
        return self.download_audio_only(url)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download YouTube videos")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument(
        "-r",
        "--resolution",
        default="highest",
        help="Video resolution (e.g., 720p, 1080p) or 'highest'",
    )
    parser.add_argument(
        "-a", "--audio-only", action="store_true", help="Download audio only"
    )
    parser.add_argument(
        "-i",
        "--info",
        action="store_true",
        help="Only show video information without downloading",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        default="downloads",
        help="Directory to save the downloaded file",
    )

    args = parser.parse_args()
    Logger.log(True, f"Getting information for: {args.url}")

    downloader = YouTubeDownloader(output_path=args.output_path)

    if args.info:
        Logger.log(True, f"Getting information for: {args.url}")
        info = downloader.get_video_info(args.url)
        if info["success"]:
            Logger.log(True, "Video information retrieved")
            print(f"\n✓ Video information retrieved successfully:")
            print(f"  Title: {info['title']}")
            print(f"  Author: {info['author']}")
            print(f"  Length: {info['length']} seconds")
            print(f"  Thumbnail: {info['thumbnail_url']}")
            print(
                f"  Available resolutions: {', '.join(info['available_resolutions'])}"
            )
            print(f"  Views: {info['views']}")
        else:
            Logger.log(False, "Failed to get video information")
            print(f"\n✗ Failed to get video info: {info['error']}")
    elif args.audio_only:
        Logger.log(True, f"Downloading audio from: {args.url}")
        result = downloader.download_audio_only(args.url)
        if result["success"]:
            print(f"  Title: {result['title']}")
            print(f"  Saved to: {result['file_path']}")
        else:
            # Error already printed in the download_audio_only method
            pass
    else:
        Logger.log(True, f"Downloading video from: {args.url}")
        result = downloader.download(args.url)
        if result["success"]:
            print(f"  Title: {result['title']}")
            print(f"  Saved to: {result['file_path']}")
        else:
            # Error already printed in the download method
            pass
