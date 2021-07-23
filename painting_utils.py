import math
import mediapipe as mp
from typing import List, Tuple, Union, Set, NamedTuple

import cv2
import dataclasses
import numpy as np

from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions.hands import HandLandmark

mp_hands = mp.solutions.hands

PRESENCE_THRESHOLD = 0.5
RGB_CHANNELS = 3
RED_COLOR = (0, 0, 255)
BLUE_VIOLET = (138, 43, 226)
VISIBILITY_THRESHOLD = 0.5


@dataclasses.dataclass
class DrawingSpec:
    # Color for drawing the annotation. Default to the green color.
    color: Tuple[int, int, int] = (0, 255, 0)
    # Thickness for drawing the annotation. Default to 2 pixels.
    thickness: int = 2
    # Circle radius. Default to 2 pixels.
    circle_radius: int = 2


def normalized_to_pixel_coordinates(
        normalized_x: float, normalized_y: float, image_width: int,
        image_height: int) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                          math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def angle_2p_3d(p1, p2, p3):
    # angle between p1p2 and p2p3

    v1 = np.array([ p1.x - p2.x, p1.y - p2.y, p1.z - p2.z ])
    v2 = np.array([ p3.x - p2.x, p3.y - p2.y, p3.z - p2.z ])

    v1mag = np.sqrt([ v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2] ])
    v1norm = np.array([ v1[0] / v1mag, v1[1] / v1mag, v1[2] / v1mag ])

    v2mag = np.sqrt(v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2])
    v2norm = np.array([ v2[0] / v2mag, v2[1] / v2mag, v2[2] / v2mag ])
    res = v1norm[0] * v2norm[0] + v1norm[1] * v2norm[1] + v1norm[2] * v2norm[2]
    angle_rad = np.arccos(res)

    return math.degrees(angle_rad)


def calc_finger_bent(p1, p2, p3, angle):
    return angle_2p_3d(p1, p2, p3) < angle


def is_index_finger_straight_others_bent(lm):
    angle_135 = 135
    angle_160 = 160

    index_finger_bent = calc_finger_bent(
        lm.landmark[HandLandmark.INDEX_FINGER_DIP],
        lm.landmark[HandLandmark.INDEX_FINGER_PIP],
        lm.landmark[HandLandmark.INDEX_FINGER_MCP],
        angle_135
    )
    middle_finger_bent = calc_finger_bent(
        lm.landmark[HandLandmark.MIDDLE_FINGER_DIP],
        lm.landmark[HandLandmark.MIDDLE_FINGER_PIP],
        lm.landmark[HandLandmark.MIDDLE_FINGER_MCP],
        angle_135
    )
    ring_finger_bent = calc_finger_bent(
        lm.landmark[HandLandmark.RING_FINGER_DIP],
        lm.landmark[HandLandmark.RING_FINGER_PIP],
        lm.landmark[HandLandmark.RING_FINGER_MCP],
        angle_135
    )
    pinky_finger_bent = calc_finger_bent(
        lm.landmark[HandLandmark.PINKY_DIP],
        lm.landmark[HandLandmark.PINKY_PIP],
        lm.landmark[HandLandmark.PINKY_MCP],
        angle_135
    )
    # for the thumb start at the tip
    thumb_bent = calc_finger_bent(
        lm.landmark[HandLandmark.THUMB_TIP],
        lm.landmark[HandLandmark.THUMB_IP],
        lm.landmark[HandLandmark.THUMB_MCP],
        angle_160
    )
    thumb_close_middle_finger = is_points_close(lm, HandLandmark.THUMB_TIP, HandLandmark.MIDDLE_FINGER_TIP, 2)

    return not index_finger_bent and ( middle_finger_bent or ring_finger_bent or pinky_finger_bent )


def is_index_and_middle_fingers_straight(lm):
    angle_135 = 135

    index_finger_bent = calc_finger_bent(
        lm.landmark[HandLandmark.INDEX_FINGER_DIP],
        lm.landmark[HandLandmark.INDEX_FINGER_PIP],
        lm.landmark[HandLandmark.INDEX_FINGER_MCP],
        angle_135
    )
    middle_finger_bent = calc_finger_bent(
        lm.landmark[HandLandmark.MIDDLE_FINGER_DIP],
        lm.landmark[HandLandmark.MIDDLE_FINGER_PIP],
        lm.landmark[HandLandmark.MIDDLE_FINGER_MCP],
        angle_135
    )
    ring_finger_bent = calc_finger_bent(
        lm.landmark[HandLandmark.RING_FINGER_DIP],
        lm.landmark[HandLandmark.RING_FINGER_PIP],
        lm.landmark[HandLandmark.RING_FINGER_MCP],
        angle_135
    )
    pinky_finger_bent = calc_finger_bent(
        lm.landmark[HandLandmark.PINKY_DIP],
        lm.landmark[HandLandmark.PINKY_PIP],
        lm.landmark[HandLandmark.PINKY_MCP],
        angle_135
    )

    return not index_finger_bent and not middle_finger_bent and ring_finger_bent or pinky_finger_bent


def is_points_close(landmark, p1, p2, factor):
    # index finger tip to index finger dip as a norm
    index_finger_tip = landmark.landmark[HandLandmark.INDEX_FINGER_TIP]
    index_finger_dip = landmark.landmark[HandLandmark.INDEX_FINGER_DIP]
    tip_to_dip_distance = calc_distance_3d(index_finger_tip, index_finger_dip)

    p1_landmark = landmark.landmark[p1]
    p2_landmark = landmark.landmark[p2]
    p1_to_p2_distance = calc_distance_3d(p1_landmark, p2_landmark)

    if p1_to_p2_distance / tip_to_dip_distance < factor:
        return True
    return False


def painting_mode_activated(results):
    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0]
        return is_points_close(lm, HandLandmark.INDEX_FINGER_TIP, HandLandmark.THUMB_TIP, 1.5)

        # other indicator (index finger straight, others bent)
        # return is_index_finger_straight_others_bent(lm)
    return False


def calc_distance_3d(p1, p2):
    return math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2 + (p2.z - p1.z) ** 2)


def is_flaky_detection(last_position, curr_position):
    # TODO implement
    return False


def index_finger_position(image, results):
    if results.multi_hand_landmarks:
        if image.shape[2] != RGB_CHANNELS:
            raise ValueError('Input image must contain three channel rgb data.')
        image_rows, image_cols, _ = image.shape
        landmark = results.multi_hand_landmarks[0].landmark[HandLandmark.INDEX_FINGER_TIP]
        return normalized_to_pixel_coordinates(landmark.x, landmark.y, image_cols, image_rows)


def to_readonly_rgb(image, flip: bool = True):
    if flip:
        image = cv2.flip(image, 1)
    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    return image


def to_writable_bgr(image):
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def draw_hand_annotation(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)


def draw_landmarks(
        image: np.ndarray,
        landmark_list: landmark_pb2.NormalizedLandmarkList,
        connections: List[Tuple[int, int]] = None,
        landmark_drawing_spec: DrawingSpec = DrawingSpec(color=RED_COLOR),
        connection_drawing_spec: DrawingSpec = DrawingSpec()):
    """Draws the landmarks and the connections on the image.

    Args:
      image: A three channel RGB image represented as numpy ndarray.
      landmark_list: A normalized landmark list proto message to be annotated on
        the image.
      connections: A list of landmark index tuples that specifies how landmarks to
        be connected in the drawing.
      landmark_drawing_spec: A DrawingSpec object that specifies the landmarks'
        drawing settings such as color, line thickness, and circle radius.
      connection_drawing_spec: A DrawingSpec object that specifies the
        connections' drawing settings such as color and line thickness.

    Raises:
      ValueError: If one of the followings:
        a) If the input image is not three channel RGB.
        b) If any connetions contain invalid landmark index.
    """
    if not landmark_list:
        return
    if image.shape[2] != RGB_CHANNELS:
        raise ValueError('Input image must contain three channel rgb data.')
    image_rows, image_cols, _ = image.shape
    idx_to_coordinates = {}
    for idx, landmark in enumerate(landmark_list.landmark):
        if ((landmark.HasField('visibility') and
             landmark.visibility < VISIBILITY_THRESHOLD) or
                (landmark.HasField('presence') and
                 landmark.presence < PRESENCE_THRESHOLD)):
            continue
        landmark_px = normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                      image_cols, image_rows)
        if landmark_px:
            idx_to_coordinates[idx] = landmark_px
    if connections:
        num_landmarks = len(landmark_list.landmark)
        # Draws the connections if the start and end landmarks are both visible.
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(f'Landmark index is out of range. Invalid connection '
                                 f'from landmark #{start_idx} to landmark #{end_idx}.')
            if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
                cv2.line(image, idx_to_coordinates[start_idx],
                         idx_to_coordinates[end_idx], connection_drawing_spec.color,
                         connection_drawing_spec.thickness)
    # Draws landmark points after finishing the connection lines, which is
    # aesthetically better.
    for landmark_px in idx_to_coordinates.values():
        cv2.circle(image, landmark_px, landmark_drawing_spec.circle_radius,
                   landmark_drawing_spec.color, landmark_drawing_spec.thickness)


def draw_painting(
        image: np.ndarray,
        drawing: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
        last_position_pix: Tuple[int, int] = None,
        curr_position_pix: Tuple[int, int] = None,
        connection_drawing_spec: DrawingSpec = DrawingSpec(color=BLUE_VIOLET)):
    """Draws the painting on the image.

    Args:
      image: A three channel RGB image represented as numpy ndarray.
      drawing: The drawing consisting of a list of tuples of points
      connection_drawing_spec: A DrawingSpec object that specifies the landmarks'
        drawing settings such as color, line thickness, and circle radius.
      last_position_pix: The last position of the drawing finger in pixel coordinates
      curr_position_pix: The curr position of the drawing finger in pixel coordinates
    """

    if last_position_pix and curr_position_pix:
        drawing.add((last_position_pix, curr_position_pix))

    if len(drawing) > 0:
        for connection in drawing:
            start, end = connection
            cv2.line(image, start, end, connection_drawing_spec.color, connection_drawing_spec.thickness)


def draw_text(image, text):
    cv2.putText(image, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, BLUE_VIOLET, 2, cv2.LINE_AA)


def rubber_mode_activated(results):
    if results.multi_hand_landmarks:
        return is_index_finger_straight_others_bent(results.multi_hand_landmarks[0])
    return False


def remove_lines(painting, curr_position):
    if curr_position is None:
        return
    to_be_removed = set()
    for line in painting:
        start, end = line
        if calc_distance_2d(start, curr_position) < 30 or calc_distance_2d(end, curr_position) < 30:
            to_be_removed.add(line)
    for line in to_be_removed:
        painting.remove(line)


def calc_distance_2d(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def handle_painting(image: np.ndarray,
                    results: NamedTuple,
                    painting: Set[Tuple[Tuple[int, int], Tuple[int, int]]],
                    last_position: Tuple[int, int] = None):
    curr_position = None
    if painting_mode_activated(results) and not is_flaky_detection(last_position, curr_position):
        curr_position = index_finger_position(image, results)
        draw_text(image, "painting activated")
    elif rubber_mode_activated(results):
        draw_text(image, "rubber")
        remove_lines(painting, index_finger_position(image, results))
    draw_painting(image, painting, last_position_pix=last_position, curr_position_pix=curr_position)
    return curr_position

