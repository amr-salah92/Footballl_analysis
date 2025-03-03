import numpy as np
import pandas as pd
from collections import defaultdict
import time

class FootballStatsEngine:
    def __init__(self, team_a_players=None, team_b_players=None):
        """
        Initialize the football statistics engine.
        
        Args:
            team_a_players: List of player names for team A
            team_b_players: List of player names for team B
        """
        # Team lineups
        self.team_a_players = team_a_players or []
        self.team_b_players = team_b_players or []
        
        # Initialize statistics dictionaries
        self.player_stats = self._initialize_player_stats()
        self.team_stats = self._initialize_team_stats()
        
        # Time tracking
        self.start_time = None
        self.current_time = 0  # Match time in seconds
        
        # Performance history for trending
        self.history = {
            "player_stats": defaultdict(list),
            "team_stats": defaultdict(list),
            "timestamps": []
        }
        
        # Expected goals (xG) model parameters
        self.xg_model = {
            "distance_factor": -0.05,  # xG decreases with distance from goal
            "angle_factor": 0.8,       # xG increases with better angle
            "base_xg": 0.3             # Base xG value
        }
    
    def _initialize_player_stats(self):
        """
        Initialize statistics dictionary for all players.
        
        Returns:
            Dictionary of player statistics
        """
        stats = {}
        
        # Initialize for team A players
        for player in self.team_a_players:
            stats[player] = self._create_empty_player_stats("a")
            
        # Initialize for team B players
        for player in self.team_b_players:
            stats[player] = self._create_empty_player_stats("b")
            
        return stats
    
    def _create_empty_player_stats(self, team):
        """
        Create empty statistics dictionary for a player.
        
        Args:
            team: Team identifier ("a" or "b")
            
        Returns:
            Dictionary with initialized statistics
        """
        return {
            "team": team,
            "distance_covered": 0.0,    # Total distance in meters
            "sprints": 0,               # Number of sprints
            "top_speed": 0.0,           # Maximum speed in m/s
            "avg_speed": 0.0,           # Average speed in m/s
            "possession_time": 0,       # Time with ball in seconds
            "passes": 0,                # Total passes
            "pass_accuracy": 0.0,       # Pass completion percentage
            "successful_passes": 0,     # Completed passes
            "key_passes": 0,            # Passes leading to shots
            "crosses": 0,               # Cross attempts
            "successful_crosses": 0,    # Successful crosses
            "long_balls": 0,            # Long pass attempts
            "successful_long_balls": 0, # Successful long passes
            "through_balls": 0,         # Through ball attempts
            "successful_through_balls": 0, # Successful through balls
            "shots": 0,                 # Total shots
            "shots_on_target": 0,       # Shots on target
            "goals": 0,                 # Goals scored
            "assists": 0,               # Goal assists
            "expected_goals": 0.0,      # Expected goals (xG)
            "dribbles": 0,              # Dribble attempts
            "successful_dribbles": 0,   # Successful dribbles
            "tackles": 0,               # Tackle attempts
            "successful_tackles": 0,    # Successful tackles
            "interceptions": 0,         # Interceptions
            "blocks": 0,                # Blocked shots/passes
            "clearances": 0,            # Clearances
            "fouls": 0,                 # Fouls committed
            "fouls_drawn": 0,           # Fouls received
            "offsides": 0,              # Offside calls
            "heatmap": np.zeros((10, 10)), # Position heatmap (10x10 grid)
            "position_history": []      # List of (x, y) positions
        }
    
    def _initialize_team_stats(self):
        """
        Initialize statistics dictionary for both teams.
        
        Returns:
            Dictionary of team statistics
        """
        return {
            "a": {
                "possession": 0.0,          # Ball possession percentage
                "shots": 0,                 # Total shots
                "shots_on_target": 0,       # Shots on target
                "goals": 0,                 # Goals scored
                "expected_goals": 0.0,      # Expected goals (xG)
                "passes": 0,                # Total passes
                "pass_accuracy": 0.0,       # Pass completion percentage
                "successful_passes": 0,     # Completed passes
                "crosses": 0,               # Cross attempts
                "corners": 0,               # Corner kicks
                "offsides": 0,              # Offside calls
                "fouls": 0,                 # Fouls committed
                "yellow_cards": 0,          # Yellow cards
                "red_cards": 0,             # Red cards
                "tackles": 0,               # Tackle attempts
                "interceptions": 0,         # Interceptions
                "formation": "4-3-3",       # Detected formation
                "dangerous_attacks": 0,     # Dangerous attacks
                "counter_attacks": 0,       # Counter attacks
                "avg_position": np.zeros((11, 2)), # Average positions (x,y) for 11 players
            },
            "b": {
                "possession": 0.0,
                "shots": 0,
                "shots_on_target": 0,
                "goals": 0,
                "expected_goals": 0.0,
                "passes": 0,
                "pass_accuracy": 0.0,
                "successful_passes": 0,
                "crosses": 0,
                "corners": 0,
                "offsides": 0,
                "fouls": 0,
                "yellow_cards": 0,
                "red_cards": 0,
                "tackles": 0,
                "interceptions": 0,
                "formation": "4-4-2",
                "dangerous_attacks": 0,
                "counter_attacks": 0,
                "avg_position": np.zeros((11, 2)),
            }
        }
    
    def start_match(self):
        """
        Start match timing and statistics tracking.
        """
        self.start_time = time.time()
        self.current_time = 0
    
    def update_match_time(self):
        """
        Update the current match time.
        
        Returns:
            Current match time in seconds
        """
        if self.start_time is None:
            return 0
            
        self.current_time = int(time.time() - self.start_time)
        return self.current_time
        
    def get_match_time_formatted(self):
        """
        Get formatted match time (MM:SS).
        
        Returns:
            Formatted match time string
        """
        minutes = self.current_time // 60
        seconds = self.current_time % 60
        return f"{minutes:02d}:{seconds:02d}"
    
    def update_player_positions(self, player_tracks, field_dimensions):
        """
        Update player position statistics based on tracking data.
        
        Args:
            player_tracks: List of player tracking dictionaries
            field_dimensions: Tuple of (width, height) in meters
        """
        field_width, field_height = field_dimensions
        
        for track in player_tracks:
            player_name = track["player_name"]
            
            # Skip if player not in our database
            if player_name not in self.player_stats:
                continue
                
            # Extract position
            center_x, center_y = track["center"]
            
            # Normalize to field dimensions (0-1 range)
            norm_x = center_x / field_width
            norm_y = center_y / field_height
            
            # Update position history
            self.player_stats[player_name]["position_history"].append((norm_x, norm_y))
            
            # Update heatmap
            heatmap = self.player_stats[player_name]["heatmap"]
            hmap_x = min(int(norm_x * 10), 9)
            hmap_y = min(int(norm_y * 10), 9)
            heatmap[hmap_y, hmap_x] += 1
            
            # Update distance covered if we have previous positions
            pos_history = self.player_stats[player_name]["position_history"]
            if len(pos_history) > 1:
                prev_x, prev_y = pos_history[-2]
                dx = (norm_x - prev_x) * field_width
                dy = (norm_y - prev_y) * field_height
                distance = np.sqrt(dx*dx + dy*dy)
                
                # Add to total distance
                self.player_stats[player_name]["distance_covered"] += distance
                
            # Update speed statistics if available in track
            if "avg_speed" in track:
                current_speed = track["avg_speed"]
                self.player_stats[player_name]["avg_speed"] = current_speed
                
                if current_speed > self.player_stats[player_name]["top_speed"]:
                    self.player_stats[player_name]["top_speed"] = current_speed
                    
                # Count sprint if speed is high enough
                if current_speed > 7.0:  # Threshold for sprint
                    self.player_stats[player_name]["sprints"] += 1
    
    def update_team_formations(self):
        """
        Update team formation detection based on player positions.
        
        Returns:
            Dictionary with detected formations for both teams
        """
        formations = {}
        
        for team in ['a', 'b']:
            # Get all players from this team
            team_players = self.team_a_players if team == 'a' else self.team_b_players
            
            # Skip if not enough players
            if len(team_players) < 10:
                formations[team] = "Unknown"
                continue
                
            # Get latest positions for all players
            player_positions = []
            for player in team_players:
                if player in self.player_stats:
                    pos_history = self.player_stats[player]["position_history"]
                    if pos_history:
                        player_positions.append(pos_history[-1])
            
            # Skip if not enough position data
            if len(player_positions) < 10:
                formations[team] = "Unknown"
                continue
                
            # Extract y-coordinates and sort them
            y_coords = sorted([pos[1] for pos in player_positions])
            
            # Count players in each "band" (defense, midfield, attack)
            # Simplified approach - would be more sophisticated in practice
            defense_count = sum(1 for y in y_coords if y < 0.35)
            midfield_count = sum(1 for y in y_coords if 0.35 <= y < 0.65)
            attack_count = sum(1 for y in y_coords if y >= 0.65)
            # Determine formation string (excluding goalkeeper)
            formation = f"{defense_count}-{midfield_count}-{attack_count}"
            formations[team] = formation
            
            # Update in team stats
            self.team_stats[team]["formation"] = formation
            
        return formations
    
    def record_pass(self, player_name, successful=True, pass_type="normal", recipient=None):
        """
        Record a pass attempt by a player.
        
        Args:
            player_name: Name of the player making the pass
            successful: Whether the pass was completed
            pass_type: Type of pass ("normal", "cross", "long_ball", "through_ball", "key_pass")
            recipient: Name of the player receiving the pass (if successful)
        """
        if player_name not in self.player_stats:
            return
            
        player = self.player_stats[player_name]
        team = player["team"]
        
        # Update player stats
        player["passes"] += 1
        if successful:
            player["successful_passes"] += 1
        
        # Update specific pass type stats
        if pass_type == "cross":
            player["crosses"] += 1
            if successful:
                player["successful_crosses"] += 1
        elif pass_type == "long_ball":
            player["long_balls"] += 1
            if successful:
                player["successful_long_balls"] += 1
        elif pass_type == "through_ball":
            player["through_balls"] += 1
            if successful:
                player["successful_through_balls"] += 1
        elif pass_type == "key_pass":
            player["key_passes"] += 1
            
        # Update team stats
        self.team_stats[team]["passes"] += 1
        if successful:
            self.team_stats[team]["successful_passes"] += 1
            
        # Update pass accuracy percentages
        if player["passes"] > 0:
            player["pass_accuracy"] = (player["successful_passes"] / player["passes"]) * 100
            
        if self.team_stats[team]["passes"] > 0:
            self.team_stats[team]["pass_accuracy"] = (self.team_stats[team]["successful_passes"] / self.team_stats[team]["passes"]) * 100
    
    def record_shot(self, player_name, on_target=False, goal=False, position=None, blocked=False):
        """
        Record a shot attempt by a player.
        
        Args:
            player_name: Name of the player taking the shot
            on_target: Whether the shot was on target
            goal: Whether the shot resulted in a goal
            position: Tuple of (x, y) coordinates of the shot
            blocked: Whether the shot was blocked
        """
        if player_name not in self.player_stats:
            return
            
        player = self.player_stats[player_name]
        team = player["team"]
        
        # Update player stats
        player["shots"] += 1
        if on_target:
            player["shots_on_target"] += 1
        if goal:
            player["goals"] += 1
            
        # Update team stats
        self.team_stats[team]["shots"] += 1
        if on_target:
            self.team_stats[team]["shots_on_target"] += 1
        if goal:
            self.team_stats[team]["goals"] += 1
            
        # Calculate xG if position is provided
        if position:
            x, y = position
            
            # Calculate distance from goal (assuming goal at x=1.0)
            distance = ((1.0 - x) ** 2 + (0.5 - y) ** 2) ** 0.5
            
            # Calculate angle (simplistic approach)
            dx = 1.0 - x
            dy = 0.5 - y
            angle = abs(np.arctan2(dy, dx))
            angle_factor = (np.pi/2 - angle) / (np.pi/2)
            
            # Calculate xG
            xg = self.xg_model["base_xg"] + (self.xg_model["distance_factor"] * distance) + (self.xg_model["angle_factor"] * angle_factor)
            
            # Ensure xG is between 0 and 1
            xg = max(0.01, min(0.99, xg))
            
            # Update xG stats
            player["expected_goals"] += xg
            self.team_stats[team]["expected_goals"] += xg
    
    def record_dribble(self, player_name, successful=True):
        """
        Record a dribble attempt by a player.
        
        Args:
            player_name: Name of the player attempting the dribble
            successful: Whether the dribble was successful
        """
        if player_name not in self.player_stats:
            return
            
        player = self.player_stats[player_name]
        
        # Update dribble stats
        player["dribbles"] += 1
        if successful:
            player["successful_dribbles"] += 1
    
    def record_tackle(self, player_name, successful=True):
        """
        Record a tackle attempt by a player.
        
        Args:
            player_name: Name of the player attempting the tackle
            successful: Whether the tackle was successful
        """
        if player_name not in self.player_stats:
            return
            
        player = self.player_stats[player_name]
        team = player["team"]
        
        # Update player stats
        player["tackles"] += 1
        if successful:
            player["successful_tackles"] += 1
            
        # Update team stats
        self.team_stats[team]["tackles"] += 1
    
    def record_defensive_action(self, player_name, action_type):
        """
        Record a defensive action by a player.
        
        Args:
            player_name: Name of the player performing the action
            action_type: Type of defensive action ("interception", "block", "clearance")
        """
        if player_name not in self.player_stats:
            return
            
        player = self.player_stats[player_name]
        team = player["team"]
        
        # Update player stats
        if action_type == "interception":
            player["interceptions"] += 1
            self.team_stats[team]["interceptions"] += 1
        elif action_type == "block":
            player["blocks"] += 1
        elif action_type == "clearance":
            player["clearances"] += 1
    
    def record_foul(self, player_name, is_offender=True):
        """
        Record a foul involving a player.
        
        Args:
            player_name: Name of the player involved in the foul
            is_offender: Whether the player committed the foul (True) or drew it (False)
        """
        if player_name not in self.player_stats:
            return
            
        player = self.player_stats[player_name]
        team = player["team"]
        
        # Update player stats
        if is_offender:
            player["fouls"] += 1
            self.team_stats[team]["fouls"] += 1
        else:
            player["fouls_drawn"] += 1
    
    def record_offside(self, player_name):
        """
        Record an offside call against a player.
        
        Args:
            player_name: Name of the player called offside
        """
        if player_name not in self.player_stats:
            return
            
        player = self.player_stats[player_name]
        team = player["team"]
        
        # Update player and team stats
        player["offsides"] += 1
        self.team_stats[team]["offsides"] += 1
    
    def record_card(self, player_name, card_type="yellow"):
        """
        Record a card given to a player.
        
        Args:
            player_name: Name of the player receiving the card
            card_type: Type of card ("yellow" or "red")
        """
        if player_name not in self.player_stats:
            return
            
        player = self.player_stats[player_name]
        team = player["team"]
        
        # Update team stats
        if card_type.lower() == "yellow":
            self.team_stats[team]["yellow_cards"] += 1
        elif card_type.lower() == "red":
            self.team_stats[team]["red_cards"] += 1
    
    def record_corner(self, team):
        """
        Record a corner kick for a team.
        
        Args:
            team: Team identifier ("a" or "b")
        """
        if team in self.team_stats:
            self.team_stats[team]["corners"] += 1
    
    def record_possession(self, team_a_percentage):
        """
        Record ball possession statistics.
        
        Args:
            team_a_percentage: Possession percentage for team A (0-100)
        """
        self.team_stats["a"]["possession"] = team_a_percentage
        self.team_stats["b"]["possession"] = 100 - team_a_percentage
    
    def record_attack(self, team, attack_type="normal"):
        """
        Record an attack by a team.
        
        Args:
            team: Team identifier ("a" or "b")
            attack_type: Type of attack ("dangerous" or "counter")
        """
        if team in self.team_stats:
            if attack_type == "dangerous":
                self.team_stats[team]["dangerous_attacks"] += 1
            elif attack_type == "counter":
                self.team_stats[team]["counter_attacks"] += 1
    
    def update_player_possession_time(self, player_name, seconds):
        """
        Update a player's possession time.
        
        Args:
            player_name: Name of the player
            seconds: Time in seconds to add to possession time
        """
        if player_name in self.player_stats:
            self.player_stats[player_name]["possession_time"] += seconds
    
    def snapshot_stats(self):
        """
        Take a snapshot of current statistics for trend analysis.
        """
        # Update match time
        self.update_match_time()
        
        # Record timestamp
        self.history["timestamps"].append(self.current_time)
        
        # Snapshot player stats
        for player_name, stats in self.player_stats.items():
            for stat_name, value in stats.items():
                # Skip complex data structures
                if stat_name in ["heatmap", "position_history"]:
                    continue
                
                self.history["player_stats"][f"{player_name}_{stat_name}"].append(value)
        
        # Snapshot team stats
        for team, stats in self.team_stats.items():
            for stat_name, value in stats.items():
                # Skip complex data structures
                if stat_name in ["avg_position"]:
                    continue
                
                self.history["team_stats"][f"{team}_{stat_name}"].append(value)
    
    def get_player_report(self, player_name):
        """
        Generate a comprehensive report for a specific player.
        
        Args:
            player_name: Name of the player
            
        Returns:
            Dictionary with player statistics
        """
        if player_name not in self.player_stats:
            return None
            
        player = self.player_stats[player_name]
        
        # Calculate additional metrics
        dribble_success_rate = 0
        if player["dribbles"] > 0:
            dribble_success_rate = (player["successful_dribbles"] / player["dribbles"]) * 100
            
        tackle_success_rate = 0
        if player["tackles"] > 0:
            tackle_success_rate = (player["successful_tackles"] / player["tackles"]) * 100
            
        # Create report
        report = {
            "name": player_name,
            "team": "Team A" if player["team"] == "a" else "Team B",
            "distance_covered_km": round(player["distance_covered"] / 1000, 2),
            "sprints": player["sprints"],
            "top_speed_kmh": round(player["top_speed"] * 3.6, 2),  # Convert m/s to km/h
            "avg_speed_kmh": round(player["avg_speed"] * 3.6, 2),
            "possession_time_min": round(player["possession_time"] / 60, 2),
            "offensive_stats": {
                "goals": player["goals"],
                "shots": player["shots"],
                "shots_on_target": player["shots_on_target"],
                "shot_accuracy": round((player["shots_on_target"] / player["shots"] * 100) if player["shots"] > 0 else 0, 2),
                "expected_goals": round(player["expected_goals"], 2),
                "assists": player["assists"],
                "key_passes": player["key_passes"],
                "dribbles": player["dribbles"],
                "dribble_success_rate": round(dribble_success_rate, 2)
            },
            "passing_stats": {
                "passes": player["passes"],
                "successful_passes": player["successful_passes"],
                "pass_accuracy": round(player["pass_accuracy"], 2),
                "crosses": player["crosses"],
                "cross_accuracy": round((player["successful_crosses"] / player["crosses"] * 100) if player["crosses"] > 0 else 0, 2),
                "long_balls": player["long_balls"],
                "long_ball_accuracy": round((player["successful_long_balls"] / player["long_balls"] * 100) if player["long_balls"] > 0 else 0, 2),
                "through_balls": player["through_balls"],
                "through_ball_accuracy": round((player["successful_through_balls"] / player["through_balls"] * 100) if player["through_balls"] > 0 else 0, 2)
            },
            "defensive_stats": {
                "tackles": player["tackles"],
                "tackle_success_rate": round(tackle_success_rate, 2),
                "interceptions": player["interceptions"],
                "blocks": player["blocks"],
                "clearances": player["clearances"]
            },
            "discipline": {
                "fouls_committed": player["fouls"],
                "fouls_drawn": player["fouls_drawn"],
                "offsides": player["offsides"]
            }
        }
        
        return report
    
    def get_team_report(self, team):
        """
        Generate a comprehensive report for a specific team.
        
        Args:
            team: Team identifier ("a" or "b")
            
        Returns:
            Dictionary with team statistics
        """
        if team not in self.team_stats:
            return None
            
        team_stats = self.team_stats[team]
        team_name = "Team A" if team == "a" else "Team B"
        
        # Calculate additional metrics
        shot_accuracy = 0
        if team_stats["shots"] > 0:
            shot_accuracy = (team_stats["shots_on_target"] / team_stats["shots"]) * 100
            
        # Get player stats for this team
        team_players = self.team_a_players if team == "a" else self.team_b_players
        
        # Calculate team aggregates
        total_distance = 0
        total_sprints = 0
        top_performers = {}
        
        for player in team_players:
            if player in self.player_stats:
                player_stats = self.player_stats[player]
                total_distance += player_stats["distance_covered"]
                total_sprints += player_stats["sprints"]
                
                # Record top goal scorer
                if "goals" not in top_performers or player_stats["goals"] > self.player_stats[top_performers["goals"]]["goals"]:
                    top_performers["goals"] = player
                    
                # Record top assister
                if "assists" not in top_performers or player_stats["assists"] > self.player_stats[top_performers["assists"]]["assists"]:
                    top_performers["assists"] = player
                    
                # Record player with most passes
                if "passes" not in top_performers or player_stats["passes"] > self.player_stats[top_performers["passes"]]["passes"]:
                    top_performers["passes"] = player
                    
                # Record player with most tackles
                if "tackles" not in top_performers or player_stats["tackles"] > self.player_stats[top_performers["tackles"]]["tackles"]:
                    top_performers["tackles"] = player
        
        # Create report
        report = {
            "team_name": team_name,
            "formation": team_stats["formation"],
            "possession": round(team_stats["possession"], 2),
            "offensive_stats": {
                "goals": team_stats["goals"],
                "shots": team_stats["shots"],
                "shots_on_target": team_stats["shots_on_target"],
                "shot_accuracy": round(shot_accuracy, 2),
                "expected_goals": round(team_stats["expected_goals"], 2),
                "corners": team_stats["corners"],
                "dangerous_attacks": team_stats["dangerous_attacks"],
                "counter_attacks": team_stats["counter_attacks"]
            },
            "passing_stats": {
                "passes": team_stats["passes"],
                "successful_passes": team_stats["successful_passes"],
                "pass_accuracy": round(team_stats["pass_accuracy"], 2),
                "crosses": team_stats["crosses"]
            },
            "defensive_stats": {
                "tackles": team_stats["tackles"],
                "interceptions": team_stats["interceptions"]
            },
            "discipline": {
                "fouls": team_stats["fouls"],
                "yellow_cards": team_stats["yellow_cards"],
                "red_cards": team_stats["red_cards"],
                "offsides": team_stats["offsides"]
            },
            "physical_stats": {
                "total_distance_km": round(total_distance / 1000, 2),
                "total_sprints": total_sprints
            },
            "top_performers": top_performers
        }
        
        return report
    
    def export_stats_to_csv(self, filename_prefix):
        """
        Export statistics to CSV files.
        
        Args:
            filename_prefix: Prefix for output CSV files
            
        Returns:
            List of generated filenames
        """
        files = []
        
        # Export player stats
        player_data = []
        for player_name, stats in self.player_stats.items():
            # Create flat dictionary for this player
            player_dict = {"player_name": player_name, "team": "Team A" if stats["team"] == "a" else "Team B"}
            
            # Add all scalar stats
            for stat_name, value in stats.items():
                if stat_name not in ["heatmap", "position_history", "team"]:
                    player_dict[stat_name] = value
                    
            player_data.append(player_dict)
            
        # Create DataFrame and save to CSV
        if player_data:
            player_df = pd.DataFrame(player_data)
            player_filename = f"{filename_prefix}_player_stats.csv"
            player_df.to_csv(player_filename, index=False)
            files.append(player_filename)
            
        # Export team stats
        team_data = []
        for team_id, stats in self.team_stats.items():
            # Create flat dictionary for this team
            team_dict = {"team": "Team A" if team_id == "a" else "Team B"}
            
            # Add all scalar stats
            for stat_name, value in stats.items():
                if stat_name not in ["avg_position"]:
                    team_dict[stat_name] = value
                    
            team_data.append(team_dict)
            
        # Create DataFrame and save to CSV
        if team_data:
            team_df = pd.DataFrame(team_data)
            team_filename = f"{filename_prefix}_team_stats.csv"
            team_df.to_csv(team_filename, index=False)
            files.append(team_filename)
            
        # Export time series data
        if self.history["timestamps"]:
            # Export player time series
            for stat_key, values in self.history["player_stats"].items():
                parts = stat_key.split("_", 1)
                if len(parts) != 2:
                    continue
                    
                player_name, stat_name = parts
                if player_name not in self.player_stats:
                    continue
                    
                # Add to time series data frame
                if "time_series_df" not in locals():
                    time_series_df = pd.DataFrame({"timestamp": self.history["timestamps"]})
                    
                time_series_df[f"{player_name}_{stat_name}"] = values
                
            # Export team time series
            for stat_key, values in self.history["team_stats"].items():
                # Add to time series data frame
                if "time_series_df" not in locals():
                    time_series_df = pd.DataFrame({"timestamp": self.history["timestamps"]})
                    
                time_series_df[stat_key] = values
                
            # Save time series to CSV
            if "time_series_df" in locals():
                time_series_filename = f"{filename_prefix}_time_series.csv"
                time_series_df.to_csv(time_series_filename, index=False)
                files.append(time_series_filename)
                
        return files