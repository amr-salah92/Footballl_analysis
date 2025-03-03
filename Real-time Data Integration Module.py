import json
import os
import pandas as pd
import numpy as np
import time
import threading
import queue
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("football_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RealTimeIntegration")

class RealTimeIntegration:
    """
    Handles real-time integration between the video processing pipeline,
    analysis modules, and the presentation dashboard.
    """
    
    def __init__(self, match_id, video_source=None, output_dir="data/processed"):
        """
        Initialize the real-time integration module.
        
        Args:
            match_id (str): Unique identifier for the match
            video_source (str, optional): Path to video file or RTSP stream URL
            output_dir (str): Directory where processed data will be saved
        """
        self.match_id = match_id
        self.video_source = video_source
        self.output_dir = output_dir
        self.match_dir = os.path.join(output_dir, match_id)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.match_dir, exist_ok=True)
        
        # Data structures for storing processed data
        self.player_data = {}
        self.match_data = {
            "match_id": match_id,
            "home_team": "",
            "away_team": "",
            "date": datetime.now().strftime('%Y-%m-%d'),
            "venue": "",
            "score": {"home": 0, "away": 0},
            "possession": {"home": 50, "away": 50},
            "shots": {"home": 0, "away": 0},
            "shots_on_target": {"home": 0, "away": 0},
            "corners": {"home": 0, "away": 0},
            "fouls": {"home": 0, "away": 0},
            "formation": {"home": "", "away": ""},
            "events": []
        }
        
        # Communication queues
        self.detection_queue = queue.Queue()  # Receives detection results
        self.tracking_queue = queue.Queue()   # Receives tracking results
        self.analysis_queue = queue.Queue()   # Receives analysis results
        self.dashboard_queue = queue.Queue()  # Sends data to dashboard
        
        # Processing threads
        self.detection_thread = None
        self.tracking_thread = None
        self.analysis_thread = None
        self.dashboard_thread = None
        
        # Thread control flags
        self.is_running = False
        self.update_interval = 1.0  # Update interval in seconds
        
        logger.info(f"Real-time integration module initialized for match {match_id}")
    
    def configure_match(self, home_team, away_team, venue, date=None, lineup=None):
        """
        Configure the match information.
        
        Args:
            home_team (str): Name of the home team
            away_team (str): Name of the away team
            venue (str): Venue name
            date (str, optional): Match date in YYYY-MM-DD format
            lineup (dict, optional): Dictionary containing lineups for both teams
        """
        self.match_data["home_team"] = home_team
        self.match_data["away_team"] = away_team
        self.match_data["venue"] = venue
        
        if date:
            self.match_data["date"] = date
            
        if lineup:
            # Initialize player data from lineup
            for team in ["home", "away"]:
                for player in lineup.get(team, []):
                    player_id = player.get("id")
                    if player_id:
                        self.player_data[player_id] = {
                            "player_id": player_id,
                            "name": player.get("name", f"Player {player_id}"),
                            "team": "Home" if team == "home" else "Away",
                            "position": player.get("position", ""),
                            "jersey_number": player.get("jersey_number", ""),
                            "distance_covered": 0.0,
                            "top_speed": 0.0,
                            "sprints": 0,
                            "passes_completed": 0,
                            "pass_accuracy": 0.0,
                            "shots": 0,
                            "xG": 0.0,
                            "goals": 0,
                            "tackles": 0,
                            "interceptions": 0,
                            "heat_map_data": np.zeros((10, 7)).tolist(),
                            "position_history": []
                        }
        
        logger.info(f"Match configured: {home_team} vs {away_team} at {venue}")
        self._save_match_data()
    
    def update_player_data(self, player_id, updates):
        """
        Update a player's data with new information.
        
        Args:
            player_id (str): Player identifier
            updates (dict): Dictionary of values to update
        """
        if player_id in self.player_data:
            self.player_data[player_id].update(updates)
            logger.debug(f"Updated player data for {player_id}: {updates}")
        else:
            # Create new player entry if it doesn't exist
            updates["player_id"] = player_id
            if "team" not in updates:
                updates["team"] = "Unknown"
            if "name" not in updates:
                updates["name"] = f"Player {player_id}"
                
            self.player_data[player_id] = updates
            logger.info(f"Created new player data for {player_id}")
    
    def update_match_data(self, updates):
        """
        Update match data with new information.
        
        Args:
            updates (dict): Dictionary of values to update
        """
        for key, value in updates.items():
            if key in self.match_data:
                if isinstance(self.match_data[key], dict) and isinstance(value, dict):
                    # Merge dictionaries for nested data
                    self.match_data[key].update(value)
                else:
                    self.match_data[key] = value
        
        logger.debug(f"Updated match data: {updates}")
        self._save_match_data()
    
    def add_event(self, event_type, time, team, player=None, details=None):
        """
        Add a new event to the match timeline.
        
        Args:
            event_type (str): Type of event (goal, shot, foul, etc.)
            time (str): Match time when the event occurred
            team (str): Team identifier (home/away)
            player (str, optional): Player identifier involved in the event
            details (dict, optional): Additional event details
        """
        event = {
            "type": event_type,
            "time": time,
            "team": team
        }
        
        if player:
            event["player"] = player
            
        if details:
            event.update(details)
            
        self.match_data["events"].append(event)
        
        # Update related match statistics
        if event_type == "goal":
            self.match_data["score"][team] += 1
            # Also update player goals
            if player:
                for pid, pdata in self.player_data.items():
                    if pdata.get("name") == player:
                        self.player_data[pid]["goals"] += 1
                        break
        
        elif event_type == "shot":
            self.match_data["shots"][team] += 1
            if details and details.get("on_target", False):
                self.match_data["shots_on_target"][team] += 1
            # Also update player shots
            if player:
                for pid, pdata in self.player_data.items():
                    if pdata.get("name") == player:
                        self.player_data[pid]["shots"] += 1
                        if "xG" in details:
                            self.player_data[pid]["xG"] += float(details["xG"])
                        break
        
        elif event_type == "corner":
            self.match_data["corners"][team] += 1
            
        elif event_type == "foul":
            self.match_data["fouls"][team] += 1
        
        logger.info(f"Added event: {event_type} at {time}' for {team}")
        self._save_match_data()
    
    def update_player_position(self, player_id, x, y, frame_number, timestamp):
        """
        Update a player's position tracking data.
        
        Args:
            player_id (str): Player identifier
            x (float): X coordinate on the pitch (normalized 0-1)
            y (float): Y coordinate on the pitch (normalized 0-1)
            frame_number (int): Video frame number
            timestamp (float): Timestamp in seconds
        """
        if player_id in self.player_data:
            # Add to position history
            self.player_data[player_id]["position_history"].append({
                "frame": frame_number,
                "timestamp": timestamp,
                "x": x,
                "y": y
            })
            
            # Update heatmap data
            heatmap = np.array(self.player_data[player_id]["heat_map_data"])
            x_bin = min(int(x * 7), 6)  # Map x to 0-6 index
            y_bin = min(int(y * 10), 9)  # Map y to 0-9 index
            heatmap[y_bin][x_bin] += 1
            self.player_data[player_id]["heat_map_data"] = heatmap.tolist()
            
            # Calculate distance covered (simplified)
            history = self.player_data[player_id]["position_history"]
            if len(history) >= 2:
                prev = history[-2]
                dx = x - prev["x"]
                dy = y - prev["y"]
                # Convert to meters (assuming pitch is 105x68 meters)
                distance_meters = np.sqrt((dx * 105)**2 + (dy * 68)**2)
                self.player_data[player_id]["distance_covered"] += distance_meters / 1000  # Convert to km
                
                # Calculate speed (m/s)
                dt = timestamp - prev["timestamp"]
                if dt > 0:
                    speed_ms = distance_meters / dt
                    speed_kmh = speed_ms * 3.6  # Convert to km/h
                    if speed_kmh > self.player_data[player_id]["top_speed"]:
                        self.player_data[player_id]["top_speed"] = speed_kmh
                    
                    # Count sprint (simplified: >20 km/h is a sprint)
                    if speed_kmh > 20:
                        self.player_data[player_id]["sprints"] += 1
    
    def start_processing(self):
        """Start all processing threads and the real-time loop."""
        if self.is_running:
            logger.warning("Processing already running")
            return
        
        self.is_running = True
        
        # Start the update thread
        self.dashboard_thread = threading.Thread(target=self._dashboard_update_loop, daemon=True)
        self.dashboard_thread.start()
        
        logger.info("Real-time processing started")
    
    def stop_processing(self):
        """Stop all processing threads and save final data."""
        self.is_running = False
        
        # Wait for threads to finish
        if self.dashboard_thread:
            self.dashboard_thread.join(timeout=2.0)
        
        # Save final data
        self._save_player_data()
        self._save_match_data()
        
        logger.info("Real-time processing stopped")
    
    def _dashboard_update_loop(self):
        """Thread that periodically updates the dashboard data."""
        while self.is_running:
            try:
                # Save current state to files for dashboard to read
                self._save_player_data()
                self._save_match_data()
                
                # Signal dashboard that new data is available
                update_data = {
                    "timestamp": time.time(),
                    "match_id": self.match_id,
                    "has_updates": True
                }
                self.dashboard_queue.put(update_data)
                
                # Sleep for the update interval
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in dashboard update loop: {e}")
    
    def _save_player_data(self):
        """Save player data to CSV file."""
        try:
            # Convert player data to DataFrame
            rows = []
            for player_id, data in self.player_data.items():
                # Create a copy without the position history (too large)
                player_row = data.copy()
                if "position_history" in player_row:
                    player_row.pop("position_history")
                
                # Convert heat_map_data to JSON string
                player_row["heat_map_data"] = json.dumps(player_row["heat_map_data"])
                rows.append(player_row)
            
            if rows:
                df = pd.DataFrame(rows)
                output_file = os.path.join(self.match_dir, "player_stats.csv")
                df.to_csv(output_file, index=False)
                logger.debug(f"Saved player data to {output_file}")
        
        except Exception as e:
            logger.error(f"Error saving player data: {e}")
    
    def _save_match_data(self):
        """Save match data to JSON file."""
        try:
            output_file = os.path.join(self.match_dir, "match_info.json")
            with open(output_file, 'w') as f:
                json.dump(self.match_data, f, indent=2)
            logger.debug(f"Saved match data to {output_file}")
        
        except Exception as e:
            logger.error(f"Error saving match data: {e}")
    
    def calculate_team_statistics(self):
        """Calculate aggregate team statistics from player data."""
        home_players = [p for p in self.player_data.values() if p.get("team") == "Home"]
        away_players = [p for p in self.player_data.values() if p.get("team") == "Away"]
        
        # Calculate possession based on passes
        home_passes = sum(p.get("passes_completed", 0) for p in home_players)
        away_passes = sum(p.get("passes_completed", 0) for p in away_players)
        total_passes = home_passes + away_passes
        
        if total_passes > 0:
            home_possession = round((home_passes / total_passes) * 100)
            away_possession = 100 - home_possession
            
            self.match_data["possession"] = {
                "home": home_possession,
                "away": away_possession
            }
        
        # Calculate additional team statistics here as needed
        
        logger.info("Team statistics calculated")
        
    def detect_formations(self):
        """
        Analyze player positions to determine team formations.
        Simplified implementation using most common positions.
        """
        # This would use machine learning in a full implementation
        # Simplified example:
        formations = {
            "home": "4-3-3",  # Default formation
            "away": "4-4-2"   # Default formation
        }
        
        self.match_data["formation"] = formations
        logger.info(f"Formations detected: Home {formations['home']}, Away {formations['away']}")
    
    def simulate_match_for_testing(self, duration_seconds=90):
        """
        Simulate a match for testing the dashboard.
        
        Args:
            duration_seconds (int): Duration of the simulation in seconds
        """
        logger.info(f"Starting match simulation for {duration_seconds} seconds")
        
        # Example match configuration
        self.configure_match(
            home_team="FC Barcelona",
            away_team="Real Madrid",
            venue="Camp Nou",
            date=datetime.now().strftime('%Y-%m-%d')
        )
        
        # Create some players
        for i in range(1, 12):
            self.player_data[f"h{i}"] = {
                "player_id": f"h{i}",
                "name": f"Home Player {i}",
                "team": "Home",
                "position": "GK" if i == 1 else ("DEF" if i < 6 else ("MID" if i < 9 else "FWD")),
                "jersey_number": i,
                "distance_covered": 0.0,
                "top_speed": 0.0,
                "sprints": 0,
                "passes_completed": 0,
                "pass_accuracy": 0.0,
                "shots": 0,
                "xG": 0.0,
                "goals": 0,
                "tackles": 0,
                "interceptions": 0,
                "heat_map_data": np.zeros((10, 7)).tolist(),
                "position_history": []
            }
            
            self.player_data[f"a{i}"] = {
                "player_id": f"a{i}",
                "name": f"Away Player {i}",
                "team": "Away",
                "position": "GK" if i == 1 else ("DEF" if i < 6 else ("MID" if i < 9 else "FWD")),
                "jersey_number": i,
                "distance_covered": 0.0,
                "top_speed": 0.0,
                "sprints": 0,
                "passes_completed": 0,
                "pass_accuracy": 0.0,
                "shots": 0,
                "xG": 0.0,
                "goals": 0,
                "tackles": 0,
                "interceptions": 0,
                "heat_map_data": np.zeros((10, 7)).tolist(),
                "position_history": []
            }
        
        # Start the processing
        self.start_processing()
        
        # Simulate a 90-minute match in accelerated time
        match_minute = 0
        simulation_start = time.time()
        try:
            while time.time() - simulation_start < duration_seconds and self.is_running:
                current_time = time.time() - simulation_start
                match_minute = int((current_time / duration_seconds) * 90)
                
                # Simulate player movements
                for player_id in self.player_data:
                    x = np.random.random()
                    y = np.random.random()
                    frame = int(current_time * 25)  # Assuming 25 FPS
                    self.update_player_position(player_id, x, y, frame, current_time)
                    
                    # Randomly update player stats
                    if np.random.random() < 0.1:  # 10% chance each second
                        team = "home" if player_id.startswith("h") else "away"
                        if np.random.random() < 0.6:  # 60% chance for pass
                            pass_completed = np.random.random() < 0.7  # 70% completion rate
                            if pass_completed:
                                self.player_data[player_id]["passes_completed"] += 1
                            total_passes = self.player_data[player_id].get("passes_completed", 0)
                            attempts = total_passes / 0.7  # Rough estimate of attempts
                            if attempts > 0:
                                self.player_data[player_id]["pass_accuracy"] = (total_passes / attempts) * 100
                                
                        elif np.random.random() < 0.1:  # 10% chance for shot
                            xg = np.random.random() * 0.5  # Random xG between 0 and 0.5
                            self.add_event("shot", str(match_minute), team, 
                                          player=self.player_data[player_id]["name"],
                                          details={"xG": xg, "on_target": np.random.random() < 0.4})
                            
                        elif np.random.random() < 0.05:  # 5% chance for goal
                            xg = 0.2 + np.random.random() * 0.6  # Higher xG for goals
                            self.add_event("goal", str(match_minute), team, 
                                          player=self.player_data[player_id]["name"],
                                          details={"xG": xg})
                            
                        elif np.random.random() < 0.1:  # 10% chance for tackle
                            self.player_data[player_id]["tackles"] += 1
                            
                        elif np.random.random() < 0.1:  # 10% chance for interception
                            self.player_data[player_id]["interceptions"] += 1
                
                # Update team statistics occasionally
                if match_minute % 5 == 0:
                    self.calculate_team_statistics()
                    
                    # Occasionally add team events
                    if np.random.random() < 0.2:
                        team = "home" if np.random.random() < 0.5 else "away"
                        self.add_event("corner", str(match_minute), team)
                        
                    if np.random.random() < 0.3:
                        team = "home" if np.random.random() < 0.5 else "away"
                        self.add_event("foul", str(match_minute), team)
                
                # Update formations halfway through
                if match_minute == 45:
                    self.detect_formations()
                    
                # Sleep to control simulation speed
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            logger.info("Simulation interrupted")
        finally:
            self.stop_processing()
            logger.info("Simulation complete")


# Example usage
if __name__ == "__main__":
    # Create integration module
    rt = RealTimeIntegration(match_id="test_match_001")
    
    # Run a test simulation
    rt.simulate_match_for_testing(duration_seconds=30)