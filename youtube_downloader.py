"""
YouTube Downloader Module

This module provides functionality to download YouTube videos using wget and yt-dlp.
With a focus on downloading audio-only from YouTube videos.
"""

import os
import json
import subprocess
import shutil
import re

from urllib.parse import urlparse, parse_qs


class YouTubeDownloader:
    """A class to handle downloading videos from YouTube using wget."""

    def __init__(self, output_path="downloads"):
        """
        Initialize the YouTubeDownloader.

        Args:
            output_path (str): Directory where videos will be saved. Defaults to "downloads".
        """
        self.output_path = output_path
        # Create the output directory if it doesn't exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # Check if required tools are installed
        self._check_requirements()

    def _check_requirements(self):
        """Check if required tools (wget and yt-dlp) are installed."""
        self.has_wget = shutil.which("wget") is not None
        if not self.has_wget:
            print("Note: wget is not installed. Will use yt-dlp for downloads instead.")
            print("To install wget (optional): brew install wget (macOS) or apt-get install wget (Ubuntu/Debian)")
        
        if not shutil.which("yt-dlp"):
            print("Error: yt-dlp is not installed. This module requires yt-dlp.")
            print("Please install it using: pip install yt-dlp")
            raise ImportError("yt-dlp is required but not installed")

    def _sanitize_filename(self, filename):
        """
        Sanitize a filename by removing invalid characters.
        
        Args:
            filename (str): The filename to sanitize
            
        Returns:
            str: Sanitized filename
        """
        # Replace invalid characters with underscore
        return re.sub(r'[\\/*?:"<>|]', "_", filename)

    def _get_video_id(self, url):
        """
        Extract the video ID from a YouTube URL.
        
        Args:
            url (str): YouTube video URL
            
        Returns:
            str: YouTube video ID or None if not found
        """
        parsed_url = urlparse(url)
        
        if parsed_url.netloc in ('youtu.be', 'www.youtu.be'):
            return parsed_url.path[1:]
        
        if parsed_url.netloc in ('youtube.com', 'www.youtube.com'):
            query_params = parse_qs(parsed_url.query)
            return query_params.get('v', [None])[0]
        
        return None

    def get_video_info(self, url):
        """
        Get information about a YouTube video using yt-dlp.
        
        Args:
            url (str): YouTube video URL
            
        Returns:
            dict: Information about the video or error details
        """
        try:
            # Use yt-dlp to get video info
            cmd = ["yt-dlp", "--dump-json", url]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Parse JSON output
            info = json.loads(result.stdout)
            
            # Extract available formats/resolutions
            formats = info.get('formats', [])
            available_resolutions = []
            for fmt in formats:
                if fmt.get('height') and fmt.get('width'):
                    resolution = f"{fmt['height']}p"
                    if resolution not in available_resolutions:
                        available_resolutions.append(resolution)
            
            return {
                "success": True,
                "title": info.get('title', 'Unknown'),
                "author": info.get('uploader', 'Unknown'),
                "length": info.get('duration', 0),
                "thumbnail_url": info.get('thumbnail', ''),
                "available_resolutions": sorted(available_resolutions, key=lambda x: int(x[:-1])),
                "views": info.get('view_count', 0),
                "rating": info.get('average_rating', 0)
            }
        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "error": f"Error getting video info: {e.stderr}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def download_video(self, url, resolution="highest"):
        """
        Download a YouTube video using wget.

        Args:
            url (str): YouTube video URL
            resolution (str): Desired resolution. Use "highest" for highest resolution or
                            a specific resolution like "720p". Defaults to "highest".

        Returns:
            dict: A dictionary containing:
                - success (bool): True if download was successful, False otherwise
                - file_path (str): Path to the downloaded file if successful
                - error (str): Error message if download failed
        """
        try:
            # Get video info first
            info = self.get_video_info(url)
            if not info["success"]:
                return info
            
            print(f"\nVideo title: {info['title']}")
            print(f"Author: {info['author']}")
            print(f"Length: {info['length']} seconds")
            print(f"Thumbnail: {info['thumbnail_url']}")
            
            # Prepare yt-dlp command to get direct URL
            height = None
            if resolution != "highest" and resolution.endswith("p"):
                height = resolution[:-1]  # Remove the 'p' to get just the number
                
            # Build format selection string for yt-dlp
            format_selector = "bestvideo+bestaudio/best"
            if height:
                if height in [res[:-1] for res in info["available_resolutions"]]:
                    format_selector = f"bestvideo[height<={height}]+bestaudio/best[height<={height}]"
                else:
                    print(f"Resolution {resolution} not available. Using highest available resolution.")
            
            # Get direct URL using yt-dlp
            cmd = ["yt-dlp", "-f", format_selector, "-g", url]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            direct_url = result.stdout.strip()
            
            # If multiple URLs are returned (video+audio), use the first one (video)
            if "\n" in direct_url:
                direct_url = direct_url.split("\n")[0]
            
            # Create safe filename
            safe_title = self._sanitize_filename(info['title'])
            output_file = os.path.join(self.output_path, f"{safe_title}.mp4")
            
            # Download the video
            print(f"Selected format: {resolution}")
            print(f"Downloading to: {output_file}")
            
            if self.has_wget:
                # Use wget if available
                wget_cmd = [
                    "wget", 
                    "-O", output_file,
                    "--progress=bar:force",
                    direct_url
                ]
                subprocess.run(wget_cmd, check=True)
            else:
                # Use yt-dlp's built-in download functionality as fallback
                ytdlp_cmd = [
                    "yt-dlp", 
                    "-f", format_selector,
                    "-o", output_file,
                    "--no-playlist",
                    url
                ]
                subprocess.run(ytdlp_cmd, check=True)
            
            return {
                "success": True,
                "file_path": output_file,
                "title": info['title']
            }
            
        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "error": f"Error: {e.stderr if hasattr(e, 'stderr') else str(e)}",
                "file_path": None
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error: {str(e)}",
                "file_path": None
            }

    def download_audio_only(self, url):
        """
        Download only the audio from a YouTube video using wget.

        Args:
            url (str): YouTube video URL

        Returns:
            dict: A dictionary containing:
                - success (bool): True if download was successful, False otherwise
                - file_path (str): Path to the downloaded file if successful
                - error (str): Error message if download failed
        """
        try:
            # Get video info first
            info = self.get_video_info(url)
            if not info["success"]:
                return info
            
            print(f"\nVideo title: {info['title']}")
            print(f"Author: {info['author']}")
            print(f"Length: {info['length']} seconds")
            
            # Get direct audio URL using yt-dlp
            cmd = ["yt-dlp", "-f", "bestaudio", "-g", url]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            direct_url = result.stdout.strip()
            
            # Create safe filename
            safe_title = self._sanitize_filename(info['title'])
            output_file = os.path.join(self.output_path, f"{safe_title}.mp3")
            
            # Download the audio
            print(f"Downloading audio to: {output_file}")
            
            if self.has_wget:
                # Use wget if available
                wget_cmd = [
                    "wget", 
                    "-O", output_file,
                    "--progress=bar:force",
                    direct_url
                ]
                subprocess.run(wget_cmd, check=True)
            else:
                # Use yt-dlp's built-in download functionality as fallback
                ytdlp_cmd = [
                    "yt-dlp", 
                    "-f", "bestaudio",
                    "-o", output_file,
                    "--no-playlist",
                    url
                ]
                subprocess.run(ytdlp_cmd, check=True)
            
            return {
                "success": True,
                "file_path": output_file,
                "title": info['title']
            }
            
        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "error": f"Error: {e.stderr if hasattr(e, 'stderr') else str(e)}",
                "file_path": None
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error: {str(e)}",
                "file_path": None
            }
            
    def download(self, url):
        """
        Main download method that always downloads only audio from YouTube videos.
        
        Args:
            url (str): YouTube video URL
            
        Returns:
            dict: A dictionary containing download results from download_audio_only
        """
        return self.download_audio_only(url)


# Example usage via command line
if __name__ == "__main__":
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Download YouTube videos")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("-r", "--resolution", default="highest", 
                        help="Video resolution (e.g., 720p, 1080p) or 'highest'")
    parser.add_argument("-a", "--audio-only", action="store_true", 
                        help="Download audio only")
    parser.add_argument("-i", "--info", action="store_true",
                        help="Only show video information without downloading")
    parser.add_argument("-o", "--output-path", default="downloads", 
                        help="Directory to save the downloaded file")
    
    args = parser.parse_args()
    
    downloader = YouTubeDownloader(output_path=args.output_path)
    
    if args.info:
        print(f"Getting information for: {args.url}")
        info = downloader.get_video_info(args.url)
        if info["success"]:
            print(f"\nTitle: {info['title']}")
            print(f"Author: {info['author']}")
            print(f"Length: {info['length']} seconds")
            print(f"Thumbnail: {info['thumbnail_url']}")
            print(f"Available resolutions: {', '.join(info['available_resolutions'])}")
            print(f"Views: {info['views']}")
        else:
            print(f"Failed to get video info: {info['error']}")
    elif args.audio_only:
        print(f"Downloading audio from: {args.url}")
        result = downloader.download_audio_only(args.url)
        if result["success"]:
            print(f"\nDownload successful!")
            print(f"Title: {result['title']}")
            print(f"Saved to: {result['file_path']}")
        else:
            print(f"\nDownload failed: {result['error']}")
    else:
        # Use the new default behavior to download audio only
        print(f"Downloading audio from: {args.url}")
        result = downloader.download(args.url)
        if result["success"]:
            print(f"\nDownload successful!")
            print(f"Title: {result['title']}")
            print(f"Saved to: {result['file_path']}")
        else:
            print(f"\nDownload failed: {result['error']}")
