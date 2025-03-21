#!/usr/bin/env python3
"""
Test script for the YouTube downloader using yt-dlp and wget.
"""

from youtube_downloader import YouTubeDownloader

def main():
    # Initialize the downloader
    downloader = YouTubeDownloader(output_path="downloads")
    
    # Example YouTube URL
    video_url = "https://youtu.be/jwfP92kaW_U?si=YFOZtBAGMZtaYAl6"
    
    # Get video information
    print("Getting video information...")
    info = downloader.get_video_info(video_url)
    
    if info["success"]:
        print(f"Video title: {info['title']}")
        print(f"Available resolutions: {info['available_resolutions']}")
        
        # Download options
        print("\nDownload options:")
        print("1. Download video in highest resolution")
        print("2. Download video in 720p (if available)")
        print("3. Download audio only")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == "1":
            print("\nDownloading video in highest resolution...")
            result = downloader.download_video(video_url)
        elif choice == "2":
            print("\nDownloading video in 720p...")
            result = downloader.download_video(video_url, resolution="720p")
        elif choice == "3":
            print("\nDownloading audio only...")
            result = downloader.download_audio_only(video_url)
        else:
            print("Invalid choice. Exiting.")
            return
        
        if result["success"]:
            print(f"\nDownload completed successfully!")
            print(f"File saved to: {result['file_path']}")
        else:
            print(f"\nDownload failed: {result['error']}")
    else:
        print(f"Failed to get video information: {info['error']}")

if __name__ == "__main__":
    main()
