import cv2
import numpy as np
from ultralytics import YOLO

class FootballObjectDetector:
    def __init__(self, player_model_path, ball_model_path, confidence_threshold=0.5):
        """
        Initialize the football object detector with YOLOv8 models.
        
        Args:
            player_model_path: Path to the YOLOv8 model for player detection
            ball_model_path: Path to the YOLOv8 model for ball detection
            confidence_threshold: Minimum confidence score to consider a detection valid
        """
        self.player_model = YOLO(player_model_path)
        self.ball_model = YOLO(ball_model_path)
        self.confidence_threshold = confidence_threshold
        
        # Class mappings for player model
        self.player_classes = {
            0: "player_team_a",
            1: "player_team_b",
            2: "referee",
            3: "goalkeeper_team_a",
            4: "goalkeeper_team_b"
        }
        
    def detect_players(self, frame):
        """
        Detect players in the given frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            List of dictionaries containing player detections with bounding boxes,
            confidence scores, and team classification
        """
        results = self.player_model(frame, verbose=False)[0]
        detections = []
        
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = result
            
            if confidence < self.confidence_threshold:
                continue
                
            class_name = self.player_classes.get(int(class_id), "unknown")
            
            detections.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": float(confidence),
                "class": class_name,
                "team": class_name.split("_")[-1] if "team" in class_name else "official"
            })
            
        return detections
    
    def detect_ball(self, frame):
        """
        Detect the ball in the given frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            Dictionary containing ball detection with bounding box and confidence score,
            or None if no ball is detected
        """
        results = self.ball_model(frame, verbose=False)[0]
        
        if len(results.boxes.data) == 0:
            return None
            
        # Get the detection with highest confidence
        best_detection = max(results.boxes.data.tolist(), key=lambda x: x[4])
        x1, y1, x2, y2, confidence, _ = best_detection
        
        if confidence < self.confidence_threshold:
            return None
            
        return {
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "confidence": float(confidence),
            "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)]
        }
    
    def process_frame(self, frame):
        """
        Process a single frame to detect players and ball.
        
        Args:
            frame: Input image frame
            
        Returns:
            Tuple of (player_detections, ball_detection)
        """
        player_detections = self.detect_players(frame)
        ball_detection = self.detect_ball(frame)
        
        return player_detections, ball_detection
        
    def visualize_detections(self, frame, player_detections, ball_detection):
        """
        Visualize detections on the frame.
        
        Args:
            frame: Input image frame
            player_detections: List of player detection dictionaries
            ball_detection: Ball detection dictionary or None
            
        Returns:
            Frame with visualized detections
        """
        output_frame = frame.copy()
        
        # Draw player detections
        for detection in player_detections:
            x1, y1, x2, y2 = detection["bbox"]
            team = detection["team"]
            
            # Set color based on team
            if team == "a":
                color = (0, 0, 255)  # Red for team A
            elif team == "b":
                color = (255, 0, 0)  # Blue for team B
            else:
                color = (0, 255, 255)  # Yellow for officials
                
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(output_frame, f"{detection['class']} {detection['confidence']:.2f}", 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw ball detection
        if ball_detection:
            x1, y1, x2, y2 = ball_detection["bbox"]
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(output_frame, f"Ball {ball_detection['confidence']:.2f}", 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return output_frame