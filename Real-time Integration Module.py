import os
import logging
import json
import csv
import time
import cv2
import numpy as np
from datetime import datetime

# Set up logging
logger = logging.getLogger("RealTimeIntegration")

class RealTimeIntegration:
    """
    Module for handling real-time integration of football analysis data,
    including data storage, visualization, and communication with external systems.
    """
    
    def __init__(self, match_id, video_source=None, output_dir="data/processed"):
        """
        Initialize the real-time integration module.
        
        Args:
            match_id (str): Unique identifier for the match
            video_source (str, optional): Video file or RTSP stream URL
            output_dir (str): Directory for saving processed data
        """
        self.match_id = match_id
        self.video_source = video_source
        self.output_dir = output_dir
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize storage for match data
        self.match_info = {}
        self.frames_metadata = []
        self.events = []
        self.player_tracks = {}
        self.team_stats = {"home": {}, "away": {}}
        
        # Initialize video writer for highlights
        self.highlight_writer = None
        self.is_recording_highlight = False
        self.highlight_buffer = []
        self.max_highlight_buffer = 150  # 5 seconds at 30fps
        
        # Initialize WebSocket server for real-time updates (placeholder)
        # self.websocket_server = None
        
        # Initialize database connection (placeholder)
        # self.db_connection = None
        
        logger.info(f"Real-time integration initialized for match: {match_id}")
    
    def configure_match(self, home_team, away_team, venue, date):
        """
        Configure match information.
        
        Args:
            home_team (str): Name of the home team
            away_team (str): Name of the away team
            venue (str): Match venue
            date (str): Match date in YYYY-MM-DD format
        """
        self.match_info = {
            "match_id": self.match_id,
            "home_team": home_team,
            "away_team": away_team,
            "venue": venue,
            "date": date,
            "start_time": datetime.now().strftime('%H:%M:%S'),
            "status": "in_progress"
        }
        
        # Initialize team statistics
        for key in ["possession", "shots", "shots_on_target", "passes", "pass_accuracy", 
                   "fouls", "yellow_cards", "red_cards", "offsides", "corners"]:
            self.team_stats["home"][key] = 0
            self.team_stats["away"][key] = 0
        
        # Save initial match information
        self._save_match_info()
        
        logger.info(f"Match configured: {home_team} vs {away_team}")
    
    def update(self, frame, detections, tracks, poses, events):
        """
        Update with the latest frame processing results.
        
        Args:
            frame (np.array): Current video frame
            detections (list): List of player and ball detections
            tracks (list): List of tracked players
            poses (list): List of player pose estimations
            events (list): List of detected events
        """
        frame_number = len(self.frames_metadata)
        timestamp = time.time()
        
        # Store minimal frame metadata
        frame_meta = {
            "frame_number": frame_number,
            "timestamp": timestamp,
            "num_detections": len(detections),
            "num_tracks": len(tracks)
        }
        self.frames_metadata.append(frame_meta)
        
        # Process and store tracking data
        for track in tracks:
            # In a real implementation, track would have player ID, position, etc.
            # Here we're just creating placeholder data
            track_id = track.get("id", f"unknown_{frame_number}")
            
            if track_id not in self.player_tracks:
                self.player_tracks[track_id] = []
                
            # Store position data
            self.player_tracks[track_id].append({
                "frame": frame_number,
                "timestamp": timestamp,
                "position": track.get("position", [0, 0]),
                "team": track.get("team", "unknown")
            })
        
        # Process and store events
        for event in events:
            event_with_time = {
                "frame": frame_number,
                "timestamp": timestamp,
                **event
            }
            self.events.append(event_with_time)
            
            # Update team statistics based on event
            self._update_stats(event)
            
            # Check if this event is highlight-worthy
            if self._is_highlight_event(event):
                self._start_highlight_recording(frame, event)
        
        # Update highlight buffer
        if self.is_recording_highlight:
            self._update_highlight(frame)
        
        # Periodically save data
        if frame_number % 300 == 0:  # Every 300 frames (10 seconds at 30fps)
            self._save_incremental_data()
            
        # Broadcast real-time updates (placeholder)
        # self._broadcast_updates(frame_meta, events)
    
    def _update_stats(self, event):
        """
        Update team statistics based on events.
        
        Args:
            event (dict): Event data
        """
        event_type = event.get("type")
        team = event.get("team")
        
        if not team or team not in ["home", "away"]:
            return
            
        if event_type == "possession_change":
            # Update possession percentage
            duration = event.get("duration", 0)
            self.team_stats[team]["possession"] += duration
        elif event_type == "shot":
            self.team_stats[team]["shots"] += 1
            if event.get("on_target", False):
                self.team_stats[team]["shots_on_target"] += 1
        elif event_type == "pass":
            self.team_stats[team]["passes"] += 1
            if event.get("completed", False):
                # Track successful passes for pass accuracy calculation
                if "successful_passes" not in self.team_stats[team]:
                    self.team_stats[team]["successful_passes"] = 0
                self.team_stats[team]["successful_passes"] += 1
                
                # Calculate pass accuracy
                total_passes = self.team_stats[team]["passes"]
                successful = self.team_stats[team]["successful_passes"]
                if total_passes > 0:
                    self.team_stats[team]["pass_accuracy"] = round(successful / total_passes * 100, 1)
        elif event_type == "foul":
            self.team_stats[team]["fouls"] += 1
        elif event_type == "card":
            card_type = event.get("card_type")
            if card_type == "yellow":
                self.team_stats[team]["yellow_cards"] += 1
            elif card_type == "red":
                self.team_stats[team]["red_cards"] += 1
        elif event_type == "offside":
            self.team_stats[team]["offsides"] += 1
        elif event_type == "corner":
            self.team_stats[team]["corners"] += 1
    
    def _is_highlight_event(self, event):
        """
        Determine if an event is worthy of being included in highlights.
        
        Args:
            event (dict): Event data
            
        Returns:
            bool: True if event should be in highlights
        """
        highlight_events = [
            "goal", "shot", "save", "card", "foul", "penalty"
        ]
        
        # Check if event type is a highlight-worthy event
        if event.get("type") in highlight_events:
            # For shots, only include those on target or close misses
            if event.get("type") == "shot":
                return event.get("on_target", False) or event.get("quality", 0) > 0.7
            
            # Include all other highlight events
            return True
            
        return False
    
    def _start_highlight_recording(self, frame, event):
        """
        Start recording a highlight clip.
        
        Args:
            frame (np.array): Current video frame
            event (dict): Event data
        """
        # If already recording, extend current highlight
        if self.is_recording_highlight:
            # Extend recording duration
            logger.debug("Extending current highlight recording")
            return
            
        logger.info(f"Starting highlight recording for event: {event.get('type')}")
        self.is_recording_highlight = True
        self.highlight_buffer = []
        
        # Initialize highlight video writer if needed
        if not self.highlight_writer:
            highlight_dir = os.path.join(self.output_dir, "highlights")
            os.makedirs(highlight_dir, exist_ok=True)
            
            height, width = frame.shape[:2]
            highlight_path = os.path.join(
                highlight_dir, 
                f"{self.match_id}_highlight_{int(time.time())}.mp4"
            )
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.highlight_writer = cv2.VideoWriter(
                highlight_path, fourcc, 30, (width, height)
            )
    
    def _update_highlight(self, frame):
        """
        Update the highlight buffer with the current frame.
        
        Args:
            frame (np.array): Current video frame
        """
        # Add frame to buffer
        self.highlight_buffer.append(frame.copy())
        
        # If buffer exceeds maximum size, write oldest frame
        if len(self.highlight_buffer) > self.max_highlight_buffer:
            if self.highlight_writer:
                self.highlight_writer.write(self.highlight_buffer.pop(0))
                
        # Check if we should stop recording
        if len(self.highlight_buffer) >= self.max_highlight_buffer:
            self._stop_highlight_recording()
    
    def _stop_highlight_recording(self):
        """Stop recording the current highlight clip."""
        if not self.is_recording_highlight:
            return
            
        logger.info("Stopping highlight recording")
        
        # Write remaining frames in buffer
        if self.highlight_writer:
            for frame in self.highlight_buffer:
                self.highlight_writer.write(frame)
                
        # Reset state
        self.is_recording_highlight = False
        self.highlight_buffer = []
    
    def _save_match_info(self):
        """Save match information to file."""
        match_info_path = os.path.join(self.output_dir, f"{self.match_id}_info.json")
        
        with open(match_info_path, 'w') as f:
            json.dump(self.match_info, f, indent=2)
    
    def _save_incremental_data(self):
        """Save incremental data to files."""
        # Save team statistics
        stats_path = os.path.join(self.output_dir, f"{self.match_id}_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(self.team_stats, f, indent=2)
            
        # Save recent events
        events_path = os.path.join(self.output_dir, f"{self.match_id}_events.json")
        with open(events_path, 'w') as f:
            json.dump(self.events[-100:] if len(self.events) > 100 else self.events, f, indent=2)
    
    def finalize(self):
        """Finalize processing and save all data."""
        logger.info("Finalizing match data processing")
        
        # Update match status
        self.match_info["status"] = "completed"
        self.match_info["end_time"] = datetime.now().strftime('%H:%M:%S')
        
        # Calculate final statistics
        total_possession = (
            self.team_stats["home"]["possession"] + 
            self.team_stats["away"]["possession"]
        )
        if total_possession > 0:
            self.team_stats["home"]["possession"] = round(
                self.team_stats["home"]["possession"] / total_possession * 100, 1
            )
            self.team_stats["away"]["possession"] = round(
                self.team_stats["away"]["possession"] / total_possession * 100, 1
            )
        
        # Save match info with final status
        self._save_match_info()
        
        # Save complete dataset
        self._save_complete_dataset()
        
        # Close highlight writer
        if self.is_recording_highlight:
            self._stop_highlight_recording()
            
        if self.highlight_writer:
            self.highlight_writer.release()
            self.highlight_writer = None
    
    def _save_complete_dataset(self):
        """Save the complete dataset."""
        # Create directory for complete dataset
        complete_dir = os.path.join(self.output_dir, "complete")
        os.makedirs(complete_dir, exist_ok=True)
        
        # Save match information
        match_path = os.path.join(complete_dir, f"{self.match_id}_match.json")
        with open(match_path, 'w') as f:
            json.dump(self.match_info, f, indent=2)
            
        # Save team statistics
        stats_path = os.path.join(complete_dir, f"{self.match_id}_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(self.team_stats, f, indent=2)
            
        # Save all events
        events_path = os.path.join(complete_dir, f"{self.match_id}_events.json")
        with open(events_path, 'w') as f:
            json.dump(self.events, f, indent=2)
            
        # Save player tracking data (potentially large)
        tracks_path = os.path.join(complete_dir, f"{self.match_id}_tracks.json")
        with open(tracks_path, 'w') as f:
            json.dump(self.player_tracks, f, indent=2)
            
        logger.info(f"Complete dataset saved to {complete_dir}")
    
    def export_data(self, output_format="json", output_path=None):
        """
        Export processed match data in specified format.
        
        Args:
            output_format (str): Export format (json, csv, etc.)
            output_path (str): Path to save exported data
            
        Returns:
            str: Path to exported data
        """
        if not output_path:
            output_path = os.path.join(
                self.output_dir, 
                "exports", 
                f"{self.match_id}_export.{output_format}"
            )
            
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Prepare export data
        export_data = {
            "match_info": self.match_info,
            "team_stats": self.team_stats,
            "events": self.events,
            # Don't include full tracking data by default as it can be huge
            "summary": {
                "total_frames": len(self.frames_metadata),
                "total_events": len(self.events),
                "tracked_players": len(self.player_tracks)
            }
        }
        
        # Export according to requested format
        if output_format.lower() == "json":
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
        elif output_format.lower() == "csv":
            # Export events as CSV
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Write header
                if self.events:
                    writer.writerow(self.events[0].keys())
                    
                    # Write data
                    for event in self.events:
                        writer.writerow(event.values())
        else:
            logger.error(f"Unsupported export format: {output_format}")
            return None
            
        logger.info(f"Data exported to {output_path}")
        return output_path