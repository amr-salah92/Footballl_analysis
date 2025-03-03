import numpy as np
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict

class FootballPlayerTracker:
    def __init__(self, max_age=30, n_init=3, nn_budget=100, lineup_data=None):
        """
        Initialize the football player tracker with DeepSORT.
        
        Args:
            max_age: Maximum number of frames to keep a track alive without detection
            n_init: Minimum number of frames for track confirmation
            nn_budget: Maximum size of the appearance descriptors gallery
            lineup_data: Dictionary mapping team to list of player names
        """
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            nn_budget=nn_budget
        )
        
        self.tracks = {}
        self.player_stats = defaultdict(lambda: {
            "positions": [],
            "distances": 0.0,
            "speeds": [],
            "possession_time": 0,
            "passes": 0,
            "shots": 0
        })
        
        # Team lineups if provided
        self.lineup_data = lineup_data or {
            "a": [f"Player A{i}" for i in range(1, 12)],
            "b": [f"Player B{i}" for i in range(1, 12)]
        }
        
        # Map track IDs to player names
        self.track_to_player = {}
        
        # Tracking metrics
        self.frame_count = 0
        self.fps = 25  # Assumed frames per second
        self.pitch_width_m = 105  # Standard pitch width in meters
        self.pitch_height_m = 68  # Standard pitch height in meters
        self.meters_per_pixel_x = None
        self.meters_per_pixel_y = None
        
    def update(self, frame, detections, pitch_corners=None):
        """
        Update tracker with new detections from a frame.
        
        Args:
            frame: Current video frame
            detections: List of player detections from detector
            pitch_corners: Optional four corner points of the pitch for coordinate conversion
            
        Returns:
            List of tracks with player information
        """
        self.frame_count += 1
        
        # Set up coordinate conversion if pitch corners provided
        if pitch_corners is not None and self.meters_per_pixel_x is None:
            self.setup_coordinate_conversion(frame.shape[1], frame.shape[0], pitch_corners)
        
        # Format detections for DeepSORT
        detection_list = []
        
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            confidence = detection["confidence"]
            team = detection["team"]
            
            detection_list.append([
                x1, y1, x2, y2, confidence, team
            ])
        
        # Update the tracker
        tracks = self.tracker.update_tracks(detection_list, frame=frame)
        
        # Process and update track information
        current_tracks = []
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            x1, y1, x2, y2 = track.to_ltrb()
            team = track.det_class
            
            # Calculate center point
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Assign player name if not already assigned
            if track_id not in self.track_to_player:
                team_idx = len([t for t in self.track_to_player.values() if t.startswith(f"Player {team.upper()}")])
                if team_idx < len(self.lineup_data[team]):
                    self.track_to_player[track_id] = self.lineup_data[team][team_idx]
                else:
                    self.track_to_player[track_id] = f"Player {team.upper()}{team_idx+1}"
            
            player_name = self.track_to_player[track_id]
            
            # Store position history
            if self.meters_per_pixel_x is not None:
                # Convert to pitch coordinates
                pitch_x = center_x * self.meters_per_pixel_x
                pitch_y = center_y * self.meters_per_pixel_y
                
                # Calculate distance if we have previous positions
                if track_id in self.player_stats and len(self.player_stats[track_id]["positions"]) > 0:
                    prev_x, prev_y = self.player_stats[track_id]["positions"][-1]
                    distance = np.sqrt((pitch_x - prev_x)**2 + (pitch_y - prev_y)**2)
                    self.player_stats[track_id]["distances"] += distance
                    
                    # Calculate speed (m/s)
                    if self.fps > 0:
                        speed = distance * self.fps
                        self.player_stats[track_id]["speeds"].append(speed)
                
                self.player_stats[track_id]["positions"].append((pitch_x, pitch_y))
            
            current_track = {
                "id": track_id,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "center": [int(center_x), int(center_y)],
                "team": team,
                "player_name": player_name
            }
            
            # Add statistics if available
            if track_id in self.player_stats:
                stats = self.player_stats[track_id]
                current_track["distance"] = round(stats["distances"], 2)
                current_track["avg_speed"] = round(np.mean(stats["speeds"]) if stats["speeds"] else 0, 2)
                current_track["max_speed"] = round(max(stats["speeds"]) if stats["speeds"] else 0, 2)
                current_track["possession_time"] = stats["possession_time"]
                current_track["passes"] = stats["passes"]
                current_track["shots"] = stats["shots"]
            
            current_tracks.append(current_track)
        
        return current_tracks
    
    def setup_coordinate_conversion(self, frame_width, frame_height, pitch_corners):
        """
        Setup conversion from pixel coordinates to physical pitch coordinates.
        
        Args:
            frame_width: Width of the video frame in pixels
            frame_height: Height of the video frame in pixels
            pitch_corners: Four corner points of the pitch in the frame
        """
        # Calculate maximum width and height in pixels
        max_width_px = max(
            np.linalg.norm(np.array(pitch_corners[1]) - np.array(pitch_corners[0])),
            np.linalg.norm(np.array(pitch_corners[3]) - np.array(pitch_corners[2]))
        )
        
        max_height_px = max(
            np.linalg.norm(np.array(pitch_corners[2]) - np.array(pitch_corners[0])),
            np.linalg.norm(np.array(pitch_corners[3]) - np.array(pitch_corners[1]))
        )
        
        # Calculate meters per pixel
        self.meters_per_pixel_x = self.pitch_width_m / max_width_px
        self.meters_per_pixel_y = self.pitch_height_m / max_height_px
    
    def update_ball_possession(self, player_tracks, ball_detection, possession_radius=10):
        """
        Update ball possession statistics based on proximity.
        
        Args:
            player_tracks: List of player tracks
            ball_detection: Ball detection information
            possession_radius: Radius in pixels to consider ball possession
        """
        if not ball_detection or not player_tracks:
            return
        
        ball_center = ball_detection["center"]
        ball_x, ball_y = ball_center
        
        closest_player = None
        min_distance = float('inf')
        
        for track in player_tracks:
            player_x, player_y = track["center"]
            distance = np.sqrt((ball_x - player_x)**2 + (ball_y - player_y)**2)
            
            if distance < min_distance:
                min_distance = distance
                closest_player = track
        
        # Update possession if ball is close enough to a player
        if closest_player and min_distance < possession_radius:
            player_id = closest_player["id"]
            self.player_stats[player_id]["possession_time"] += 1
    
    def detect_passes_and_shots(self, player_tracks, ball_detection, goal_areas):
        """
        Detect passes and shots based on ball movement and player positions.
        
        Args:
            player_tracks: List of player tracks
            ball_detection: Current ball detection
            goal_areas: Tuple of goal area bounding boxes ((team_a_x1, y1, x2, y2), (team_b_x1, y1, x2, y2))
        """
        # Implementation would track ball trajectory and detect when ball moves between players
        # or toward goal areas, updating the player_stats accordingly
        pass
    
    def visualize_tracks(self, frame, tracks, ball_detection=None, show_stats=True):
        """
        Visualize player tracks and statistics on the frame.
        
        Args:
            frame: Input image frame
            tracks: List of player track dictionaries
            ball_detection: Ball detection dictionary or None
            show_stats: Whether to show player statistics
            
        Returns:
            Frame with visualized tracks
        """
        output_frame = frame.copy()
        
        # Draw player tracks
        for track in tracks:
            x1, y1, x2, y2 = track["bbox"]
            team = track["team"]
            player_name = track["player_name"]
            
            # Set color based on team
            if team == "a":
                color = (0, 0, 255)  # Red for team A
            elif team == "b":
                color = (255, 0, 0)  # Blue for team B
            else:
                color = (0, 255, 255)  # Yellow for officials
                
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
            
            # Display player information
            info_text = player_name
            cv2.putText(output_frame, info_text, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Display statistics if requested
            if show_stats and "distance" in track:
                stats_y = y2 + 15
                cv2.putText(output_frame, f"Dist: {track['distance']}m", 
                           (x1, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                           
                stats_y += 15
                cv2.putText(output_frame, f"Speed: {track['avg_speed']}m/s", 
                           (x1, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                           
                stats_y += 15
                cv2.putText(output_frame, f"Poss: {track['possession_time']}", 
                           (x1, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw ball detection
        if ball_detection:
            x1, y1, x2, y2 = ball_detection["bbox"]
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
        return output_frame
    
    def generate_heatmap(self, frame_shape, player_id=None, team=None):
        """
        Generate position heatmap for a player or team.
        
        Args:
            frame_shape: Shape of the video frame (height, width)
            player_id: Specific player ID to generate heatmap for, or None for all specified team
            team: Team to generate heatmap for if player_id is None
            
        Returns:
            Heatmap image
        """
        # Implementation would create a 2D histogram of player positions and visualize it as a heatmap
        pass