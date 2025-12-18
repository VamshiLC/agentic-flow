"""
License Plate Tracker

Tracks plates across video frames and uses voting to get the best OCR reading.

Features:
- IoU-based matching to track same plate across frames
- Collects multiple OCR readings per plate
- Voting system to determine most likely plate text
- Filters out inconsistent/noisy detections
"""

from typing import List, Dict, Tuple, Optional
from collections import Counter
import logging

logger = logging.getLogger(__name__)


def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]

    Returns:
        IoU score (0.0 to 1.0)
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


class PlateTracker:
    """
    Tracks license plates across video frames.

    Uses IoU matching to identify same plate in consecutive frames,
    collects all OCR readings, and uses voting to determine best text.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        min_readings: int = 2,
        max_frames_missing: int = 5
    ):
        """
        Initialize plate tracker.

        Args:
            iou_threshold: Minimum IoU to consider same plate (default 0.3)
            min_readings: Minimum OCR readings needed for valid plate
            max_frames_missing: Max frames a plate can be missing before removing
        """
        self.iou_threshold = iou_threshold
        self.min_readings = min_readings
        self.max_frames_missing = max_frames_missing

        # Tracked plates: {track_id: {...plate_info...}}
        self.tracked_plates: Dict[int, Dict] = {}
        self.next_track_id = 1
        self.current_frame = 0

    def update(self, frame_idx: int, detections: List[Dict]) -> List[Dict]:
        """
        Update tracker with new frame detections.

        Args:
            frame_idx: Current frame index
            detections: List of plate detections from current frame

        Returns:
            List of detections with track_id assigned
        """
        self.current_frame = frame_idx

        # Mark all existing tracks as not updated this frame
        for track_id in self.tracked_plates:
            self.tracked_plates[track_id]['updated'] = False

        updated_detections = []

        for det in detections:
            bbox = det.get('bbox', [])
            plate_text = det.get('plate_text', 'UNREADABLE')
            confidence = det.get('ocr_confidence', 0.0)

            if len(bbox) != 4:
                continue

            # Try to match with existing tracked plate
            matched_track_id = self._find_matching_track(bbox)

            if matched_track_id is not None:
                # Update existing track
                self._update_track(matched_track_id, det, frame_idx)
                det['track_id'] = matched_track_id
            else:
                # Create new track
                track_id = self._create_track(det, frame_idx)
                det['track_id'] = track_id

            updated_detections.append(det)

        # Remove stale tracks (not seen for too many frames)
        self._remove_stale_tracks()

        return updated_detections

    def _find_matching_track(self, bbox: List[int]) -> Optional[int]:
        """Find existing track that matches this bbox."""
        best_iou = 0.0
        best_track_id = None

        for track_id, track in self.tracked_plates.items():
            if track['updated']:
                continue  # Already matched this frame

            iou = calculate_iou(bbox, track['last_bbox'])

            if iou > best_iou and iou >= self.iou_threshold:
                best_iou = iou
                best_track_id = track_id

        return best_track_id

    def _create_track(self, detection: Dict, frame_idx: int) -> int:
        """Create new track for a plate."""
        track_id = self.next_track_id
        self.next_track_id += 1

        plate_text = detection.get('plate_text', 'UNREADABLE')
        confidence = detection.get('ocr_confidence', 0.0)

        self.tracked_plates[track_id] = {
            'track_id': track_id,
            'first_frame': frame_idx,
            'last_frame': frame_idx,
            'last_bbox': detection.get('bbox', []),
            'ocr_readings': [],
            'updated': True,
            'frames_missing': 0
        }

        # Add OCR reading if valid
        if plate_text != 'UNREADABLE' and confidence > 0.3:
            self.tracked_plates[track_id]['ocr_readings'].append({
                'text': plate_text,
                'confidence': confidence,
                'frame': frame_idx
            })

        return track_id

    def _update_track(self, track_id: int, detection: Dict, frame_idx: int):
        """Update existing track with new detection."""
        track = self.tracked_plates[track_id]

        track['last_frame'] = frame_idx
        track['last_bbox'] = detection.get('bbox', [])
        track['updated'] = True
        track['frames_missing'] = 0

        # Add OCR reading if valid
        plate_text = detection.get('plate_text', 'UNREADABLE')
        confidence = detection.get('ocr_confidence', 0.0)

        if plate_text != 'UNREADABLE' and confidence > 0.3:
            track['ocr_readings'].append({
                'text': plate_text,
                'confidence': confidence,
                'frame': frame_idx
            })

    def _remove_stale_tracks(self):
        """Remove tracks that haven't been seen for too long."""
        stale_ids = []

        for track_id, track in self.tracked_plates.items():
            if not track['updated']:
                track['frames_missing'] += 1

                if track['frames_missing'] > self.max_frames_missing:
                    stale_ids.append(track_id)

        for track_id in stale_ids:
            del self.tracked_plates[track_id]

    def get_best_plate_text(self, track_id: int) -> Tuple[str, float]:
        """
        Get best plate text for a track using voting.

        Args:
            track_id: Track ID

        Returns:
            Tuple of (best_text, confidence)
        """
        if track_id not in self.tracked_plates:
            return 'UNREADABLE', 0.0

        track = self.tracked_plates[track_id]
        readings = track['ocr_readings']

        if len(readings) == 0:
            return 'UNREADABLE', 0.0

        # Count occurrences of each text
        text_counts = Counter(r['text'] for r in readings)

        # Get most common text
        most_common = text_counts.most_common(1)[0]
        best_text = most_common[0]
        count = most_common[1]

        # Calculate confidence based on vote percentage
        vote_confidence = count / len(readings)

        # Also consider average OCR confidence for this text
        avg_ocr_conf = sum(
            r['confidence'] for r in readings if r['text'] == best_text
        ) / count

        # Combined confidence
        final_confidence = (vote_confidence + avg_ocr_conf) / 2

        return best_text, final_confidence

    def get_all_tracked_plates(self) -> List[Dict]:
        """
        Get summary of all tracked plates with best readings.

        Returns:
            List of plate summaries with voting results
        """
        results = []

        for track_id, track in self.tracked_plates.items():
            readings = track['ocr_readings']

            # Skip tracks with too few readings
            if len(readings) < self.min_readings:
                continue

            best_text, confidence = self.get_best_plate_text(track_id)

            if best_text == 'UNREADABLE':
                continue

            # Get all unique readings for this track
            text_counts = Counter(r['text'] for r in readings)

            results.append({
                'track_id': track_id,
                'plate_text': best_text,
                'confidence': round(confidence, 2),
                'total_readings': len(readings),
                'vote_count': text_counts[best_text],
                'first_frame': track['first_frame'],
                'last_frame': track['last_frame'],
                'last_bbox': track['last_bbox'],
                'all_readings': dict(text_counts)
            })

        return results

    def reset(self):
        """Reset tracker for new video."""
        self.tracked_plates = {}
        self.next_track_id = 1
        self.current_frame = 0
