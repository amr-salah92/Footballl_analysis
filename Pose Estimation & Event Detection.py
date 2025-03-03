import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque

class PoseEstimationSystem:
    def __init__(self, confidence_threshold=0.5):
        """
        Initialize the pose estimation system.
        
        Args:
            confidence_threshold: Minimum confidence score to consider a pose keypoint valid
        """
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=confidence_threshold,
            min_tracking_confidence=confidence_threshold
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize action recognition model
        self.action_model = self._load_action_recognition_model()
        
        # List of possible football actions
        self.action_classes = [
            "running", "walking", "standing", "jumping", "sliding",
            "kicking", "passing", "shooting", "tackling", "goalkeeper_save"
        ]
        
        # Pose sequence buffer for action recognition
        self.pose_history = {}  # Player ID -> deque of poses
        self.sequence_length = 15  # Number of frames to consider for action
        
        # Action detection cooldown to prevent rapid oscillation
        self.action_cooldown = {}  # Player ID -> {action: cooldown_frames}
        
    def _load_action_recognition_model(self):
        """
        Load pre-trained action recognition model.
        
        Returns:
            TensorFlow model for action recognition
        """
        try:
            # Placeholder for actual model loading
            # In production, replace with:
            # return tf.keras.models.load_model('path_to_model')
            
            # Mock model for demonstration
            class MockActionModel:
                def predict(self, pose_sequence):
                    # Return random predictions for demonstration
                    return np.random.random(10)
                    
            return MockActionModel()
        except Exception as e:
            print(f"Warning: Could not load action model: {e}")
            return None
    
    def process_frame(self, frame, player_detections):
        """
        Process a video frame and detect poses for all players.
        
        Args:
            frame: Video frame as numpy array
            player_detections: List of player detection dictionaries with
                              'player_id', 'bbox', and 'team' keys
        
        Returns:
            Dictionary mapping player IDs to pose data
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = {}
        
        for player in player_detections:
            player_id = player['player_id']
            x1, y1, x2, y2 = player['bbox']
            
            # Ensure valid coordinates
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
            
            # Skip if bounding box is too small
            if (x2 - x1) < 10 or (y2 - y1) < 10:
                continue
                
            # Extract player crop
            player_crop = frame_rgb[y1:y2, x1:x2]
            if player_crop.size == 0:
                continue
                
            # Process pose
            pose_results = self.pose.process(player_crop)
            
            # Skip if no pose detected
            if not pose_results.pose_landmarks:
                continue
                
            # Normalize landmarks to the player's bounding box
            normalized_landmarks = []
            for landmark in pose_results.pose_landmarks.landmark:
                # Convert relative coordinates to absolute frame coordinates
                px = x1 + landmark.x * (x2 - x1)
                py = y1 + landmark.y * (y2 - y1)
                confidence = landmark.visibility
                normalized_landmarks.append((px, py, confidence))
            
            # Create player pose entry
            results[player_id] = {
                'landmarks': normalized_landmarks,
                'confidence': np.mean([lm[2] for lm in normalized_landmarks]),
                'bbox': (x1, y1, x2, y2),
                'team': player['team']
            }
            
            # Update pose history for action recognition
            if player_id not in self.pose_history:
                self.pose_history[player_id] = deque(maxlen=self.sequence_length)
            
            # Add flattened landmarks to history
            flat_landmarks = np.array(normalized_landmarks).flatten()
            self.pose_history[player_id].append(flat_landmarks)
        
        return results
    
    def recognize_actions(self):
        """
        Recognize actions for players based on pose history.
        
        Returns:
            Dictionary mapping player IDs to detected actions
        """
        if not self.action_model:
            return {}
            
        results = {}
        
        for player_id, pose_sequence in self.pose_history.items():
            # Skip if we don't have enough frames
            if len(pose_sequence) < self.sequence_length:
                continue
                
            # Prepare input for action model
            X = np.array([list(pose_sequence)])
            
            # Get action predictions
            predictions = self.action_model.predict(X)
            
            # Get top action
            action_idx = np.argmax(predictions)
            action = self.action_classes[action_idx]
            confidence = predictions[action_idx]
            
            # Apply cooldown to prevent action flicker
            if player_id in self.action_cooldown:
                if action in self.action_cooldown[player_id] and self.action_cooldown[player_id][action] > 0:
                    # Skip this action if in cooldown
                    self.action_cooldown[player_id][action] -= 1
                    continue
            else:
                self.action_cooldown[player_id] = {}
                
            # Set cooldown for this action
            self.action_cooldown[player_id][action] = 10  # 10 frames cooldown
                
            # Record the detected action
            results[player_id] = {
                'action': action,
                'confidence': float(confidence)
            }
        
        return results
    
    def analyze_biomechanics(self, player_poses):
        """
        Analyze biomechanical efficiency of player movements.
        
        Args:
            player_poses: Dictionary of player poses from process_frame
            
        Returns:
            Dictionary of biomechanical metrics by player ID
        """
        biomechanics = {}
        
        for player_id, pose_data in player_poses.items():
            landmarks = pose_data['landmarks']
            
            # Skip if not enough landmarks
            if len(landmarks) != 33:  # MediaPipe returns 33 landmarks
                continue
                
            # Extract key points
            # Convert landmarks to numpy array for easier calculation
            landmarks_arr = np.array(landmarks)
            
            # Calculate joint angles
            try:
                # Hip angles (simplified)
                left_hip_angle = self._calculate_angle(
                    landmarks_arr[23],  # Left hip
                    landmarks_arr[25],  # Left knee
                    landmarks_arr[27]   # Left ankle
                )
                
                right_hip_angle = self._calculate_angle(
                    landmarks_arr[24],  # Right hip
                    landmarks_arr[26],  # Right knee
                    landmarks_arr[28]   # Right ankle
                )
                
                # Knee angles
                left_knee_angle = self._calculate_angle(
                    landmarks_arr[23],  # Left hip
                    landmarks_arr[25],  # Left knee
                    landmarks_arr[27]   # Left ankle
                )
                
                right_knee_angle = self._calculate_angle(
                    landmarks_arr[24],  # Right hip
                    landmarks_arr[26],  # Right knee
                    landmarks_arr[28]   # Right ankle
                )
                
                # Shoulder angles
                left_shoulder_angle = self._calculate_angle(
                    landmarks_arr[11],  # Left shoulder
                    landmarks_arr[13],  # Left elbow
                    landmarks_arr[15]   # Left wrist
                )
                
                right_shoulder_angle = self._calculate_angle(
                    landmarks_arr[12],  # Right shoulder
                    landmarks_arr[14],  # Right elbow
                    landmarks_arr[16]   # Right wrist
                )
                
                # Calculate posture metrics
                spine_alignment = self._calculate_spine_alignment(landmarks_arr)
                
                # Store biomechanics data
                biomechanics[player_id] = {
                    'joint_angles': {
                        'left_hip': left_hip_angle,
                        'right_hip': right_hip_angle,
                        'left_knee': left_knee_angle,
                        'right_knee': right_knee_angle,
                        'left_shoulder': left_shoulder_angle,
                        'right_shoulder': right_shoulder_angle
                    },
                    'posture': {
                        'spine_alignment': spine_alignment
                    },
                    'efficiency_score': self._calculate_efficiency_score(
                        left_hip_angle, right_hip_angle,
                        left_knee_angle, right_knee_angle,
                        spine_alignment
                    )
                }
            except Exception as e:
                # Skip biomechanics for this player if calculation fails
                print(f"Biomechanics calculation error for player {player_id}: {e}")
                continue
        
        return biomechanics
    
    def _calculate_angle(self, a, b, c):
        """
        Calculate angle between three points.
        
        Args:
            a, b, c: Points as (x, y, confidence) tuples, with b as the vertex
            
        Returns:
            Angle in degrees
        """
        a = np.array([a[0], a[1]])
        b = np.array([b[0], b[1]])
        c = np.array([c[0], c[1]])
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def _calculate_spine_alignment(self, landmarks_arr):
        """
        Calculate spine alignment score.
        
        Args:
            landmarks_arr: Numpy array of landmarks
            
        Returns:
            Alignment score (0-100, higher is better)
        """
        # Extract spine keypoints
        nose = landmarks_arr[0, :2]
        shoulders_mid = (landmarks_arr[11, :2] + landmarks_arr[12, :2]) / 2
        hips_mid = (landmarks_arr[23, :2] + landmarks_arr[24, :2]) / 2
        
        # Calculate alignment
        v1 = shoulders_mid - nose
        v2 = hips_mid - shoulders_mid
        
        # Normalize vectors
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        
        # Calculate dot product to find alignment
        alignment = np.dot(v1_norm, v2_norm)
        
        # Convert to score (0-100)
        score = (alignment + 1) * 50  # Scale from [-1, 1] to [0, 100]
        
        return float(score)
    
    def _calculate_efficiency_score(self, left_hip, right_hip, left_knee, right_knee, spine_alignment):
        """
        Calculate overall movement efficiency score.
        
        Args:
            Various joint angles and posture metrics
            
        Returns:
            Efficiency score (0-100)
        """
        # Optimal angles (simplified model)
        optimal_running_angles = {
            'hip': 165,
            'knee': 155
        }
        
        # Calculate deviations
        hip_deviation = (abs(left_hip - optimal_running_angles['hip']) + 
                         abs(right_hip - optimal_running_angles['hip'])) / 2
        
        knee_deviation = (abs(left_knee - optimal_running_angles['knee']) + 
                          abs(right_knee - optimal_running_angles['knee'])) / 2
        
        # Calculate efficiency scores for each component
        hip_score = max(0, 100 - hip_deviation)
        knee_score = max(0, 100 - knee_deviation)
        posture_score = spine_alignment
        
        # Weighted average
        efficiency_score = 0.4 * hip_score + 0.4 * knee_score + 0.2 * posture_score
        
        return float(efficiency_score)
    
    def draw_pose_on_frame(self, frame, player_poses):
        """
        Draw detected poses on the frame.
        
        Args:
            frame: Video frame as numpy array
            player_poses: Dictionary of player poses from process_frame
            
        Returns:
            Frame with poses drawn
        """
        annotated_frame = frame.copy()
        
        for player_id, pose_data in player_poses.items():
            landmarks = pose_data['landmarks']
            team = pose_data.get('team', 'unknown')
            
            # Choose color based on team
            color = (0, 255, 0) if team == 'a' else (0, 0, 255)
            
            # Draw landmarks
            for i, (x, y, conf) in enumerate(landmarks):
                if conf > 0.5:
                    cv2.circle(annotated_frame, (int(x), int(y)), 5, color, -1)
            
            # Draw connections between landmarks
            landmark_pairs = [
                # Torso
                (11, 12), (12, 24), (24, 23), (23, 11),
                # Right arm
                (12, 14), (14, 16),
                # Left arm
                (11, 13), (13, 15),
                # Right leg
                (24, 26), (26, 28),
                # Left leg
                (23, 25), (25, 27)
            ]
            
            for pair in landmark_pairs:
                if (landmarks[pair[0]][2] > 0.5 and landmarks[pair[1]][2] > 0.5):
                    pt1 = (int(landmarks[pair[0]][0]), int(landmarks[pair[0]][1]))
                    pt2 = (int(landmarks[pair[1]][0]), int(landmarks[pair[1]][1]))
                    cv2.line(annotated_frame, pt1, pt2, color, 2)
            
            # Add player ID and team
            x1, y1, _, _ = pose_data['bbox']
            cv2.putText(annotated_frame, f"ID: {player_id}", (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return annotated_frame


class EventDetectionSystem:
    def __init__(self, field_dimensions=(105, 68)):
        """
        Initialize the event detection system.
        
        Args:
            field_dimensions: Tuple of (length, width) in meters
        """
        self.field_length, self.field_width = field_dimensions
        
        # Event detection thresholds
        self.thresholds = {
            'pass_distance': 5.0,         # Minimum distance for pass detection
            'shot_distance': 35.0,        # Maximum distance for shot detection
            'tackle_distance': 1.0,       # Maximum distance for tackle detection
            'ball_possession_distance': 2.0,  # Max distance to consider player possessing the ball
            'min_pass_speed': 2.0,        # Minimum ball speed for pass detection
            'min_shot_speed': 5.0,        # Minimum ball speed for shot detection
        }
        
        # Track ball possession
        self.ball_possession = {
            'player_id': None,
            'team': None,
            'duration': 0
        }
        
        # Ball trajectory buffer
        self.ball_trajectory = deque(maxlen=30)  # 1 second at 30fps
        
        # Event cooldown to prevent duplicate events
        self.event_cooldown = {
            'pass': 0,
            'shot': 0,
            'tackle': 0,
            'interception': 0
        }
        
        # Initialize football stats engine
        self.stats_engine = None
        
        # xG model parameters
        self.xg_model = {
            'distance_factor': -0.05,
            'angle_factor': 0.8,
            'base_xg': 0.3
        }
    
    def set_stats_engine(self, stats_engine):
        """
        Set stats engine to record detected events.
        
        Args:
            stats_engine: FootballStatsEngine instance
        """
        self.stats_engine = stats_engine
    
    def update(self, player_positions, ball_position, player_actions):
        """
        Update the event detection system with new tracking data.
        
        Args:
            player_positions: Dictionary mapping player IDs to (x, y) positions
            ball_position: (x, y) position of the ball
            player_actions: Dictionary of player actions from PoseEstimationSystem
            
        Returns:
            Dictionary of detected events
        """
        detected_events = []
        
        # Update ball trajectory
        if ball_position is not None:
            self.ball_trajectory.append({
                'position': ball_position,
                'timestamp': time.time()
            })
        
        # Decrement cooldowns
        for event_type in self.event_cooldown:
            if self.event_cooldown[event_type] > 0:
                self.event_cooldown[event_type] -= 1
        
        # Update ball possession
        self._update_ball_possession(player_positions, ball_position)
        
        # Detect passes
        if len(self.ball_trajectory) >= 2 and self.event_cooldown['pass'] == 0:
            pass_event = self._detect_pass(player_positions)
            if pass_event:
                detected_events.append(pass_event)
                self.event_cooldown['pass'] = 15  # Half second cooldown at 30fps
        
        # Detect shots
        if len(self.ball_trajectory) >= 2 and self.event_cooldown['shot'] == 0:
            shot_event = self._detect_shot(player_positions)
            if shot_event:
                detected_events.append(shot_event)
                self.event_cooldown['shot'] = 30  # 1 second cooldown
        
        # Detect tackles
        if self.event_cooldown['tackle'] == 0:
            tackle_event = self._detect_tackle(player_positions, player_actions)
            if tackle_event:
                detected_events.append(tackle_event)
                self.event_cooldown['tackle'] = 45  # 1.5 second cooldown
        
        # Detect interceptions
        if self.event_cooldown['interception'] == 0:
            interception_event = self._detect_interception(player_positions)
            if interception_event:
                detected_events.append(interception_event)
                self.event_cooldown['interception'] = 30  # 1 second cooldown
        
        # Record events in stats engine if available
        if self.stats_engine:
            for event in detected_events:
                self._record_event_in_stats(event)
        
        return detected_events
    
    def _update_ball_possession(self, player_positions, ball_position):
        """
        Update which player has possession of the ball.
        
        Args:
            player_positions: Dictionary mapping player IDs to (x, y) positions
            ball_position: (x, y) position of the ball
        """
        if ball_position is None:
            return
            
        # Find closest player to the ball
        closest_player = None
        closest_distance = float('inf')
        closest_team = None
        
        for player_id, position in player_positions.items():
            distance = np.linalg.norm(np.array(position) - np.array(ball_position))
            
            if distance < closest_distance:
                closest_distance = distance
                closest_player = player_id
                # Determine team (assuming player_id format includes team info)
                closest_team = 'a' if player_id.startswith('a') else 'b'
        
        # Check if closest player is close enough to be considered in possession
        if closest_distance <= self.thresholds['ball_possession_distance']:
            # Check if possession changed
            if self.ball_possession['player_id'] != closest_player:
                # Possession changed
                self.ball_possession = {
                    'player_id': closest_player,
                    'team': closest_team,
                    'duration': 0
                }
                
                # If stats engine is available, update possession time for previous player
                if self.stats_engine and self.ball_possession['player_id']:
                    self.stats_engine.update_player_possession_time(
                        self.ball_possession['player_id'], 
                        self.ball_possession['duration']
                    )
            else:
                # Same player still has possession
                self.ball_possession['duration'] += 1
        else:
            # No player has possession
            if self.ball_possession['player_id'] is not None:
                # If stats engine is available, update possession time for previous player
                if self.stats_engine:
                    self.stats_engine.update_player_possession_time(
                        self.ball_possession['player_id'], 
                        self.ball_possession['duration']
                    )
                
                self.ball_possession = {
                    'player_id': None,
                    'team': None,
                    'duration': 0
                }
    
    def _detect_pass(self, player_positions):
        """
        Detect passing events.
        
        Args:
            player_positions: Dictionary mapping player IDs to (x, y) positions
            
        Returns:
            Pass event dictionary or None
        """
        if len(self.ball_trajectory) < 3:
            return None
            
        # Get current and previous ball positions
        current = self.ball_trajectory[-1]['position']
        previous = self.ball_trajectory[-3]['position']
        
        # Calculate ball speed
        time_diff = self.ball_trajectory[-1]['timestamp'] - self.ball_trajectory[-3]['timestamp']
        if time_diff == 0:
            return None
            
        distance = np.linalg.norm(np.array(current) - np.array(previous))
        ball_speed = distance / time_diff
        
        # Check if ball is moving fast enough to be a pass
        if ball_speed < self.thresholds['min_pass_speed']:
            return None
            
        # Find the origin player (who passed the ball)
        origin_player = None
        for player_id, position in player_positions.items():
            player_to_previous = np.linalg.norm(np.array(position) - np.array(previous))
            if player_to_previous < self.thresholds['ball_possession_distance']:
                origin_player = player_id
                break
                
        if origin_player is None:
            return None
            
        # Determine the target player (recipient)
        target_player = None
        min_distance = float('inf')
        
        # Project ball trajectory
        trajectory_vector = np.array(current) - np.array(previous)
        trajectory_vector = trajectory_vector / np.linalg.norm(trajectory_vector)
        projected_position = np.array(current) + trajectory_vector * 5.0  # Project 5m ahead
        
        for player_id, position in player_positions.items():
            if player_id == origin_player:
                continue
                
            distance_to_projection = np.linalg.norm(np.array(position) - projected_position)
            if distance_to_projection < min_distance:
                min_distance = distance_to_projection
                target_player = player_id
        
        # Determine pass type
        pass_type = "normal"
        
        # Check for through ball (pass goes into space)
        if min_distance > 3.0:
            pass_type = "through_ball"
        
        # Check for long ball (pass covers large distance)
        pass_distance = np.linalg.norm(np.array(current) - np.array(previous))
        if pass_distance > 20.0:
            pass_type = "long_ball"
        
        # Check for cross (pass from wide area)
        field_width_half = self.field_width / 2
        if abs(previous[0]) > field_width_half * 0.7:  # If pass originated from wide area
            pass_type = "cross"
        
        # Create pass event
        return {
            'type': 'pass',
            'subtype': pass_type,
            'origin_player': origin_player,
            'target_player': target_player,
            'origin_position': previous,
            'target_position': current,
            'speed': ball_speed,
            'distance': pass_distance,
            'timestamp': self.ball_trajectory[-1]['timestamp']
        }
    
    def _detect_shot(self, player_positions):
        """
        Detect shooting events.
        
        Args:
            player_positions: Dictionary mapping player IDs to (x, y) positions
            
        Returns:
            Shot event dictionary or None
        """
        if len(self.ball_trajectory) < 3:
            return None
            
        # Get current and previous ball positions
        current = self.ball_trajectory[-1]['position']
        previous = self.ball_trajectory[-3]['position']
        
        # Calculate ball speed
        time_diff = self.ball_trajectory[-1]['timestamp'] - self.ball_trajectory[-3]['timestamp']
        if time_diff == 0:
            return None
            
        distance = np.linalg.norm(np.array(current) - np.array(previous))
        ball_speed = distance / time_diff
        
        # Check if ball is moving fast enough to be a shot
        if ball_speed < self.thresholds['min_shot_speed']:
            return None
            
        # Find the origin player (who shot the ball)
        origin_player = None
        for player_id, position in player_positions.items():
            player_to_previous = np.linalg.norm(np.array(position) - np.array(previous))
            if player_to_previous < self.thresholds['ball_possession_distance']:
                origin_player = player_id
                break
                
        if origin_player is None:
            return None
            
        # Check if the shot is heading toward a goal
        goal_position = (self.field_length, self.field_width/2)
        
        # Vector from previous ball position to current
        shot_vector = np.array(current) - np.array(previous)
        shot_vector = shot_vector / np.linalg.norm(shot_vector)
        
        # Vector from previous ball position to goal
        goal_vector = np.array(goal_position) - np.array(previous)
        goal_vector = goal_vector / np.linalg.norm(goal_vector)
        
        # Calculate dot product to see if vectors are aligned
        alignment = np.dot(shot_vector, goal_vector)
        
        # Check if shot is heading toward goal (alignment > 0.7 means < 45 degree angle)
        if alignment < 0.7:
            return None
            
        # Calculate expected goals (xG)
        distance_to_goal = np.linalg.norm(np.array(goal_position) - np.array(previous))
        
        # Check if within reasonable shooting distance
        if distance_to_goal > self.thresholds['shot_distance']:
            return None
            
        # Calculate angle to goal
        angle = np.arccos(alignment)
        
        # Calculate xG based on distance and angle
        xg = max(0.01, min(0.99, self.xg_model['base_xg'] + 
                           self.xg_model['distance_factor'] * distance_to_goal +
                           self.xg_model['angle_factor'] * (1 - angle/(np.pi/2))))
        
        # Create shot event
        return {
            'type': 'shot',
            'player': origin_player,
            'position': previous,
            'speed': ball_speed,
            'distance_to_goal': distance_to_goal,
            'angle': np.degrees(angle),
            'expected_goals': xg,
            'timestamp': self.ball_trajectory[-1]['timestamp']
        }
    
    def _detect_tackle(self, player_positions, player_actions):
        """
        Detect tackling events.
        
        Args:
            player_positions: Dictionary mapping player IDs to (x, y) positions
            player_actions: Dictionary of player actions from PoseEstimationSystem
            
        Returns:
            Tackle event dictionary or None
        """
        # Check for players performing tackle action
        tackling_players = []
        for player_id, action_data in player_actions.items():
            if action_data['action'] == 'tackling' and action_data['confidence'] > 0.7:
                tackling_players.append(player_id)
        
        if not tackling_players:
            return None
            
        # Find the player with the ball
        ball_carrier = self.ball_possession['player_id']
        if ball_carrier is None:
            return None
            
        # Check if any tackling player is close to the ball carrier
        for tackler_id in tackling_players:
            if tackler_id == ball_carrier:
                continue
                
            # Check if they're from different teams
            tackler_team = 'a' if tackler_id.startswith('a') else 'b'
            carrier_team = 'a' if ball_carrier.startswith('a') else 'b'
            
            if tackler_team == carrier_team:
                continue
                
            # Check if they're close enough
            tackler_pos = player_positions[tackler_id]
            carrier_pos = player_positions[ball_carrier]
            
            distance = np.linalg.norm(np.array(tackler_pos) - np.array(carrier_pos))
            
            if distance <= self.thresholds['tackle_distance']:
                # Create tackle event
                return {
                    'type': 'tackle',
                    'tackler': tackler_id,
                    'tackler_team': tackler_team,
                    'player_tackled': ball_carrier,
                    'tackled_team': carrier_team,
                    'position': carrier_pos,
                    'timestamp': time.time()
                }
                
        return None
    
    def _detect_interception(self, player_positions):
        """
        Detect interception events.
        
        Args:
            player_positions: Dictionary mapping player IDs to (x, y) positions
            
        Returns:
            Interception event dictionary or None
        """
        # Check if possession has changed between teams
        if (self.ball_possession['player_id'] is not None and 
            self.ball_possession['team'] is not None):
            
            # Get the players from the previous possession
            last_possession = getattr(self, '_last_possession', {
                'player_id': None,
                'team': None
            })
            
            if (last_possession['player_id'] is not None and 
                last_possession['team'] != self.ball_possession['team']):
                
                # Possession changed between teams - potential interception
                interceptor = self.ball_possession['player_id']
                interceptor_pos = player_positions[interceptor]
                
                # Create interception event
                interception_event = {
                    'type': 'interception',
                    'interceptor': interceptor,
                    'interceptor_team': self.ball_possession['team'],
                    'previous_team': last_possession['team'],
                    'position': interceptor_pos,
                    'timestamp': time.time()
                }
                
                # Update last possession
                self._last_possession = {
                    'player_id': self.ball_possession['player_id'],
                    'team': self.ball_possession['team']
                }
                
                return interception_event
        
        # Update last possession
        self._last_possession = {
            'player_id': self.ball_possession['player_id'],
            'team': self.ball_possession['team']
        }
        
        return None
    
    def _record_event_in_stats(self, event):
        """
        Record detected event in the stats engine.
        
        Args:
            event: Event dictionary
        """
        if not self.stats_engine:
            return
            
        event_type = event['type']
        
        if event_type == 'pass':
            self.stats_engine.record_pass(
                player_id=event['origin_player'],
                pass_type=event['subtype'],
                success=event['target_player'] is not None,
                distance=event['distance']
            )
            
        elif event_type == 'shot':
            self.stats_engine.record_shot(
                player_id=event['player'],
                distance=event['distance_to_goal'],
                xg=event['expected_goals'],
                # Determine if goal based on ball trajectory (simplified)
                on_target=event['angle'] < 15
            )
            
        elif event_type == 'tackle':
            self.stats_engine.record_tackle(
                player_id=event['tackler'],
                success=True  # We only detect successful tackles for now
            )
            
        elif event_type == 'interception':
            self.stats_engine.record_interception(
                player_id=event['interceptor']
            )
    
    def get_xg(self, shot_position):
        """
        Calculate expected goals for a shot from a given position.
        
        Args:
            shot_position: (x, y) position of the shot
            
        Returns:
            Expected goals value between 0 and 1
        """
        # Calculate distance to goal
        goal_position = (self.field_length, self.field_width/2)
        distance = np.linalg.norm(np.array(shot_position) - np.array(goal_position))
        
        # Calculate angle to goal (simplified)
        goal_width = 7.32  # Standard goal width in meters
        angle_radians = np.arctan(goal_width / (2 * distance))
        angle_factor = angle_radians / (np.pi/2)  # Normalize to [0,1]
        
        # Calculate xG based on distance and angle
        xg = max(0.01, min(0.99, self.xg_model['base_xg'] + 
                         self.xg_model['distance_factor'] * distance +
                         self.xg_model['angle_factor'] * angle_factor))
        
        return xg


class FootballStatsEngine:
    def __init__(self):
        """
        Initialize the football statistics engine.
        """
        # Player stats dictionary
        self.player_stats = {}
        
        # Team stats dictionary
        self.team_stats = {
            'a': self._initialize_team_stats(),
            'b': self._initialize_team_stats()
        }
        
        # Match stats
        self.match_stats = {
            'duration': 0,
            'possession': {'a': 0, 'b': 0},
            'shots': {'a': 0, 'b': 0},
            'shots_on_target': {'a': 0, 'b': 0},
            'goals': {'a': 0, 'b': 0},
            'passes': {'a': 0, 'b': 0},
            'pass_accuracy': {'a': 0, 'b': 0},
            'tackles': {'a': 0, 'b': 0},
            'interceptions': {'a': 0, 'b': 0},
            'fouls': {'a': 0, 'b': 0},
            'yellow_cards': {'a': 0, 'b': 0},
            'red_cards': {'a': 0, 'b': 0},
            'expected_goals': {'a': 0.0, 'b': 0.0}
        }
    
    def _initialize_team_stats(self):
        """
        Initialize stats dictionary for a team.
        
        Returns:
            Team stats dictionary
        """
        return {
            'possession_time': 0,
            'shots': 0,
            'shots_on_target': 0,
            'goals': 0,
            'passes': {
                'attempted': 0,
                'completed': 0,
                'accuracy': 0.0,
                'types': {
                    'normal': 0,
                    'through_ball': 0,
                    'long_ball': 0,
                    'cross': 0
                }
            },
            'tackles': {
                'attempted': 0,
                'successful': 0,
                'success_rate': 0.0
            },
            'interceptions': 0,
            'fouls': 0,
            'cards': {
                'yellow': 0,
                'red': 0
            },
            'expected_goals': 0.0
        }
    
    def _initialize_player_stats(self, player_id):
        """
        Initialize stats dictionary for a player.
        
        Args:
            player_id: Unique player identifier
            
        Returns:
            Player stats dictionary
        """
        # Determine team
        team = 'a' if player_id.startswith('a') else 'b'
        
        return {
            'team': team,
            'minutes_played': 0,
            'possession_time': 0,
            'distance_covered': 0.0,
            'sprints': 0,
            'top_speed': 0.0,
            'shots': 0,
            'shots_on_target': 0,
            'goals': 0,
            'assists': 0,
            'passes': {
                'attempted': 0,
                'completed': 0,
                'accuracy': 0.0,
                'types': {
                    'normal': 0,
                    'through_ball': 0,
                    'long_ball': 0,
                    'cross': 0
                }
            },
            'key_passes': 0,
            'tackles': {
                'attempted': 0,
                'successful': 0,
                'success_rate': 0.0
            },
            'interceptions': 0,
            'duels': {
                'total': 0,
                'won': 0,
                'success_rate': 0.0
            },
            'fouls': {
                'committed': 0,
                'suffered': 0
            },
            'cards': {
                'yellow': 0,
                'red': 0
            },
            'expected_goals': 0.0,
            'expected_assists': 0.0,
            'biomechanics': {
                'efficiency_scores': []
            }
        }
    
    def update_player_position(self, player_id, position, speed):
        """
        Update player position and calculate movement stats.
        
        Args:
            player_id: Unique player identifier
            position: (x, y) position on the field
            speed: Current speed in m/s
        """
        # Initialize player stats if not exists
        if player_id not in self.player_stats:
            self.player_stats[player_id] = self._initialize_player_stats(player_id)
        
        # Get player's previous position if exists
        prev_position = getattr(self, f'_prev_position_{player_id}', None)
        
        if prev_position is not None:
            # Calculate distance covered since last update
            distance = np.linalg.norm(np.array(position) - np.array(prev_position))
            self.player_stats[player_id]['distance_covered'] += distance
            
            # Update top speed if current speed is higher
            if speed > self.player_stats[player_id]['top_speed']:
                self.player_stats[player_id]['top_speed'] = speed
                
            # Count as sprint if speed is above threshold (7 m/s ~ 25 km/h)
            if speed > 7.0 and getattr(self, f'_prev_speed_{player_id}', 0) <= 7.0:
                self.player_stats[player_id]['sprints'] += 1
        
        # Store current position and speed for next update
        setattr(self, f'_prev_position_{player_id}', position)
        setattr(self, f'_prev_speed_{player_id}', speed)
        
        # Update minutes played
        self.player_stats[player_id]['minutes_played'] += 1/1800  # Assuming 30fps for 60 minutes
    
    def update_player_possession_time(self, player_id, frames):
        """
        Update possession time for a player.
        
        Args:
            player_id: Unique player identifier
            frames: Number of frames the player had possession
        """
        # Initialize player stats if not exists
        if player_id not in self.player_stats:
            self.player_stats[player_id] = self._initialize_player_stats(player_id)
            
        # Update player possession time
        self.player_stats[player_id]['possession_time'] += frames
        
        # Update team possession time
        team = self.player_stats[player_id]['team']
        self.team_stats[team]['possession_time'] += frames
        
        # Update match possession percentages
        total_possession = sum(self.team_stats[t]['possession_time'] for t in ['a', 'b'])
        if total_possession > 0:
            for team in ['a', 'b']:
                self.match_stats['possession'][team] = (
                    self.team_stats[team]['possession_time'] / total_possession * 100
                )
    
    def record_pass(self, player_id, pass_type='normal', success=True, distance=0.0):
        """
        Record a passing event.
        
        Args:
            player_id: Unique player identifier
            pass_type: Type of pass (normal, through_ball, long_ball, cross)
            success: Whether the pass was completed successfully
            distance: Pass distance in meters
        """
        # Initialize player stats if not exists
        if player_id not in self.player_stats:
            self.player_stats[player_id] = self._initialize_player_stats(player_id)
            
        # Get player's team
        team = self.player_stats[player_id]['team']
        
        # Update player pass stats
        self.player_stats[player_id]['passes']['attempted'] += 1
        if success:
            self.player_stats[player_id]['passes']['completed'] += 1
        self.player_stats[player_id]['passes']['types'][pass_type] += 1
        
        # Update pass accuracy
        attempts = self.player_stats[player_id]['passes']['attempted']
        completions = self.player_stats[player_id]['passes']['completed']
        if attempts > 0:
            self.player_stats[player_id]['passes']['accuracy'] = (
                completions / attempts * 100
            )
        
        # Update team pass stats
        self.team_stats[team]['passes']['attempted'] += 1
        if success:
            self.team_stats[team]['passes']['completed'] += 1
        self.team_stats[team]['passes']['types'][pass_type] += 1
        
        # Update team pass accuracy
        team_attempts = self.team_stats[team]['passes']['attempted']
        team_completions = self.team_stats[team]['passes']['completed']
        if team_attempts > 0:
            self.team_stats[team]['passes']['accuracy'] = (
                team_completions / team_attempts * 100
            )
            
        # Update match pass stats
        self.match_stats['passes'][team] += 1
        self.match_stats['pass_accuracy'][team] = self.team_stats[team]['passes']['accuracy']
    
    def record_shot(self, player_id, distance=0.0, xg=0.0, on_target=False, goal=False):
        """
        Record a shot event.
        
        Args:
            player_id: Unique player identifier
            distance: Shot distance in meters
            xg: Expected goals value
            on_target: Whether the shot was on target
            goal: Whether the shot resulted in a goal
        """
        # Initialize player stats if not exists
        if player_id not in self.player_stats:
            self.player_stats[player_id] = self._initialize_player_stats(player_id)
            
        # Get player's team
        team = self.player_stats[player_id]['team']
        
        # Update player shot stats
        self.player_stats[player_id]['shots'] += 1
        self.player_stats[player_id]['expected_goals'] += xg
        
        if on_target:
            self.player_stats[player_id]['shots_on_target'] += 1
            
        if goal:
            self.player_stats[player_id]['goals'] += 1
        
        # Update team shot stats
        self.team_stats[team]['shots'] += 1
        self.team_stats[team]['expected_goals'] += xg
        
        if on_target:
            self.team_stats[team]['shots_on_target'] += 1
            
        if goal:
            self.team_stats[team]['goals'] += 1
            
        # Update match shot stats
        self.match_stats['shots'][team] += 1
        self.match_stats['shots_on_target'][team] += 1 if on_target else 0
        self.match_stats['goals'][team] += 1 if goal else 0
        self.match_stats['expected_goals'][team] += xg
    
    def record_tackle(self, player_id, success=True):
        """
        Record a tackling event.
        
        Args:
            player_id: Unique player identifier
            success: Whether the tackle was successful
        """
        # Initialize player stats if not exists
        if player_id not in self.player_stats:
            self.player_stats[player_id] = self._initialize_player_stats(player_id)
            
        # Get player's team
        team = self.player_stats[player_id]['team']
        
        # Update player tackle stats
        self.player_stats[player_id]['tackles']['attempted'] += 1
        if success:
            self.player_stats[player_id]['tackles']['successful'] += 1
            
        # Update tackle success rate
        attempts = self.player_stats[player_id]['tackles']['attempted']
        successes = self.player_stats[player_id]['tackles']['successful']
        if attempts > 0:
            self.player_stats[player_id]['tackles']['success_rate'] = (
                successes / attempts * 100
            )
        
        # Update team tackle stats
        self.team_stats[team]['tackles']['attempted'] += 1
        if success:
            self.team_stats[team]['tackles']['successful'] += 1
            
        # Update team tackle success rate
        team_attempts = self.team_stats[team]['tackles']['attempted']
        team_successes = self.team_stats[team]['tackles']['successful']
        if team_attempts > 0:
            self.team_stats[team]['tackles']['success_rate'] = (
                team_successes / team_attempts * 100
            )
            
        # Update match tackle stats
        self.match_stats['tackles'][team] += 1
    
    def record_interception(self, player_id):
        """
        Record an interception event.
        
        Args:
            player_id: Unique player identifier
        """
        # Initialize player stats if not exists
        if player_id not in self.player_stats:
            self.player_stats[player_id] = self._initialize_player_stats(player_id)
            
        # Get player's team
        team = self.player_stats[player_id]['team']
        
        # Update player interception stats
        self.player_stats[player_id]['interceptions'] += 1
        
        # Update team interception stats
        self.team_stats[team]['interceptions'] += 1
        
        # Update match interception stats
        self.match_stats['interceptions'][team] += 1
    
    def record_biomechanics(self, player_id, efficiency_score):
        """
        Record biomechanical efficiency score for a player.
        
        Args:
            player_id: Unique player identifier
            efficiency_score: Movement efficiency score (0-100)
        """
        # Initialize player stats if not exists
        if player_id not in self.player_stats:
            self.player_stats[player_id] = self._initialize_player_stats(player_id)
            
        # Add efficiency score to player's biomechanics data
        self.player_stats[player_id]['biomechanics']['efficiency_scores'].append(efficiency_score)
    
    def get_player_stats(self, player_id):
        """
        Get statistics for a specific player.
        
        Args:
            player_id: Unique player identifier
            
        Returns:
            Player stats dictionary
        """
        if player_id in self.player_stats:
            # Calculate average efficiency score if available
            if self.player_stats[player_id]['biomechanics']['efficiency_scores']:
                scores = self.player_stats[player_id]['biomechanics']['efficiency_scores']
                self.player_stats[player_id]['biomechanics']['average_efficiency'] = sum(scores) / len(scores)
            
            return self.player_stats[player_id]
        else:
            return None
    
    def get_team_stats(self, team):
        """
        Get statistics for a specific team.
        
        Args:
            team: Team identifier ('a' or 'b')
            
        Returns:
            Team stats dictionary
        """
        if team in self.team_stats:
            return self.team_stats[team]
        else:
            return None
    
    def get_match_stats(self):
        """
        Get overall match statistics.
        
        Returns:
            Match stats dictionary
        """
        return self.match_stats
    
    def generate_player_report(self, player_id):
        """
        Generate a comprehensive report for a player.
        
        Args:
            player_id: Unique player identifier
            
        Returns:
            Report dictionary with key metrics and insights
        """
        if player_id not in self.player_stats:
            return None
            
        stats = self.player_stats[player_id]
        
        # Calculate key performance indicators
        pass_completion = stats['passes']['accuracy']
        goals_per_shot = stats['goals'] / stats['shots'] if stats['shots'] > 0 else 0
        tackle_success = stats['tackles']['success_rate']
        
        # Calculate average biomechanical efficiency
        avg_efficiency = 0
        if stats['biomechanics']['efficiency_scores']:
            avg_efficiency = sum(stats['biomechanics']['efficiency_scores']) / len(stats['biomechanics']['efficiency_scores'])
        
        # Generate report
        report = {
            'summary': {
                'player_id': player_id,
                'team': stats['team'],
                'minutes_played': stats['minutes_played'],
                'distance_covered': stats['distance_covered'],
                'top_speed': stats['top_speed'],
                'sprints': stats['sprints']
            },
            'offensive': {
                'goals': stats['goals'],
                'shots': stats['shots'],
                'shots_on_target': stats['shots_on_target'],
                'shot_accuracy': stats['shots_on_target'] / stats['shots'] * 100 if stats['shots'] > 0 else 0,
                'expected_goals': stats['expected_goals'],
                'goals_above_expected': stats['goals'] - stats['expected_goals'],
                'key_passes': stats['key_passes'],
                'assists': stats['assists']
            },
            'passing': {
                'total_passes': stats['passes']['attempted'],
                'completed_passes': stats['passes']['completed'],
                'pass_completion': pass_completion,
                'pass_types': stats['passes']['types']
            },
            'defensive': {
                'tackles': stats['tackles']['attempted'],
                'successful_tackles': stats['tackles']['successful'],
                'tackle_success_rate': tackle_success,
                'interceptions': stats['interceptions'],
                'duels_won': stats['duels']['won'],
                'duel_success_rate': stats['duels']['success_rate']
            },
            'biomechanics': {
                'average_efficiency': avg_efficiency,
                'efficiency_trend': 'improving' if self._calculate_efficiency_trend(stats['biomechanics']['efficiency_scores']) > 0 else 'declining'
            },
            'insights': []
        }
        
        # Generate insights
        if pass_completion > 85:
            report['insights'].append("Excellent pass accuracy, consistently finding teammates.")
        elif pass_completion < 70:
            report['insights'].append("Struggling with passing precision, may need to improve decision making.")
            
        if stats['expected_goals'] > 1 and stats['goals'] == 0:
            report['insights'].append("Underperforming in front of goal, getting into good positions but lacking finishing.")
        elif stats['goals'] > stats['expected_goals'] + 1:
            report['insights'].append("Clinical finisher, converting chances at an above-average rate.")
            
        if avg_efficiency > 85:
            report['insights'].append("Excellent movement efficiency, conserving energy and maintaining good form.")
        elif avg_efficiency < 60:
            report['insights'].append("Below average biomechanical efficiency, may be at higher risk of injury.")
            
        if stats['top_speed'] > 9.0:  # 9 m/s ~ 32 km/h
            report['insights'].append("Exceptional top speed, a real threat in transition.")
            
        if stats['interceptions'] > 5:
            report['insights'].append("Reading the game well, frequently intercepting opposition passes.")
            
        return report
    
    def _calculate_efficiency_trend(self, efficiency_scores):
        """
        Calculate the trend in efficiency scores.
        
        Args:
            efficiency_scores: List of efficiency scores
            
        Returns:
            Trend value (positive for improving, negative for declining)
        """
        if len(efficiency_scores) < 5:
            return 0
            
        # Use last 5 values for trend
        recent = efficiency_scores[-5:]
        
        # Simple linear regression
        x = np.arange(len(recent))
        y = np.array(recent)
        
        # Calculate slope
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        
        numerator = np.sum((x - mean_x) * (y - mean_y))
        denominator = np.sum((x - mean_x) ** 2)
        
        if denominator == 0:
            return 0
            
        slope = numerator / denominator
        
        return slope