"""
ROCK PAPER SCISSORS – GESTURE CONTROL
Single-file webcam game using Python, OpenCV, and MediaPipe.

Install:
  python -m pip install opencv-python==4.10.0.84 mediapipe==0.10.14 numpy==1.26.4

Run:
  python main.py

Controls: SPACE = play | R = reset score | Q/ESC = quit
"""
import random
from collections import Counter, deque

import cv2
import mediapipe as mp


class HandDetector:
    """MediaPipe hand landmark detector."""
    def __init__(self, max_hands=1, detection_confidence=0.7,
                 tracking_confidence=0.7):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            model_complexity=1,
        )

    def process(self, rgb_frame):
        return self.hands.process(rgb_frame)

    def draw_landmarks(self, frame, hand_landmarks):
        self.mp_drawing.draw_landmarks(
            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
        )

    def close(self):
        self.hands.close()


def classify_gesture(landmarks):
    """Classify the four non-thumb fingers as Rock, Paper, Scissors."""
    if landmarks is None or len(landmarks) < 21:
        return "Unknown"

    # Tip/PIP landmark pairs: index, middle, ring, little fingers.
    pairs = ((8, 6), (12, 10), (16, 14), (20, 18))
    extended = [
        landmarks[tip].y < landmarks[pip].y - 0.02
        for tip, pip in pairs
    ]

    if extended == [False, False, False, False]:
        return "Rock"
    if extended == [True, True, True, True]:
        return "Paper"
    if extended == [True, True, False, False]:
        return "Scissors"
    return "Unknown"


MOVES = ("Rock", "Paper", "Scissors")
PLAYER_WINS = {
    ("Rock", "Scissors"),
    ("Paper", "Rock"),
    ("Scissors", "Paper"),
}


def computer_move():
    return random.choice(MOVES)


def decide_winner(player, computer):
    if player == computer:
        return "Draw"
    if (player, computer) in PLAYER_WINS:
        return "You win!"
    return "Computer wins!"


def draw_text(frame, text, position, scale=0.7,
              color=(245, 245, 245), thickness=2):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def main():
    detector = HandDetector()
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not open webcam. Check camera connection/access.")
        detector.close()
        return

    player_score = computer_score = draws = 0
    recent = deque(maxlen=7)
    player_move, cpu_move = "—", "—"
    result, last_gesture = "Show a gesture, then press SPACE", "Waiting..."

    print("SPACE = play | R = reset | Q/ESC = quit")

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Could not read webcam frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detection = detector.process(rgb)

            if detection.multi_hand_landmarks:
                hand = detection.multi_hand_landmarks[0]
                detector.draw_landmarks(frame, hand)
                gesture = classify_gesture(hand.landmark)
                if gesture != "Unknown":
                    recent.append(gesture)
                    last_gesture = gesture
                else:
                    recent.clear()
                    last_gesture = "Unclear pose"
            else:
                recent.clear()
                last_gesture = "No hand detected"

            stable = "Hold a gesture steady"
            if len(recent) == recent.maxlen:
                candidate, count = Counter(recent).most_common(1)[0]
                if count >= 5:
                    stable = candidate

            cv2.rectangle(frame, (0, 0), (frame.shape[1], 58), (30, 30, 30), -1)
            draw_text(frame, "ROCK  PAPER  SCISSORS", (18, 38), 0.85)
            draw_text(frame, f"Detected: {last_gesture}", (20, 100), 0.68, (80, 230, 255))
            draw_text(frame, f"Ready: {stable}", (20, 135), 0.62, (100, 255, 150))
            draw_text(frame, f"You: {player_move}", (20, 205), 0.75)
            draw_text(frame, f"Computer: {cpu_move}", (20, 245), 0.75)
            draw_text(frame, f"Result: {result}", (20, 290), 0.72, (100, 255, 150))
            draw_text(
                frame,
                f"Score  You {player_score} | Computer {computer_score} | Draws {draws}",
                (20, frame.shape[0] - 55), 0.62
            )
            draw_text(frame, "SPACE: play   R: reset   Q/ESC: quit",
                      (20, frame.shape[0] - 20), 0.52, (210, 210, 210))

            cv2.imshow("Rock Paper Scissors - Gesture Control", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            elif key == ord("r"):
                player_score = computer_score = draws = 0
                player_move, cpu_move = "—", "—"
                result = "Score reset"
                recent.clear()
            elif key == 32:
                if stable in MOVES:
                    player_move = stable
                    cpu_move = computer_move()
                    result = decide_winner(player_move, cpu_move)
                    if result == "You win!":
                        player_score += 1
                    elif result == "Computer wins!":
                        computer_score += 1
                    else:
                        draws += 1
                    recent.clear()
                else:
                    result = "Hold a clear gesture before pressing SPACE"
    finally:
        camera.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
