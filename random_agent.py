import gym, cv2
import numpy as np

# Creating the Environment
cliffEnv = gym.make("CliffWalking-v0")

# Handy functions for Visuals
def initialize_frame():
    width, height = 600, 200
    img = np.ones(shape=(height, width, 3)) * 255.0
    margin_horizontal = 6
    margin_vertical = 2

    for i in range(13):  # Vertical lines
        img = cv2.line(img, (49 * i + margin_horizontal, margin_vertical),
                       (49 * i + margin_horizontal, 200 - margin_vertical), (0, 0, 0), 1)

    for i in range(5):  # Horizontal lines
        img = cv2.line(img, (margin_horizontal, 49 * i + margin_vertical),
                       (600 - margin_horizontal, 49 * i + margin_vertical), (0, 0, 0), 1)

    # Cliff area
    img = cv2.rectangle(img, (49 * 1 + margin_horizontal + 2, 49 * 3 + margin_vertical + 2),
                        (49 * 11 + margin_horizontal - 2, 49 * 4 + margin_vertical - 2), (255, 0, 255), -1)
    img = cv2.putText(img, "Cliff", (49 * 5 + margin_horizontal, 49 * 4 + margin_vertical - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Goal
    img = cv2.putText(img, "G", (49 * 11 + margin_horizontal + 10, 49 * 4 + margin_vertical - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    return img

def put_agent(img, state):
    margin_horizontal = 6
    margin_vertical = 2
    row, column = np.unravel_index(state, (4, 12))
    cv2.putText(img, "A", (49 * column + margin_horizontal + 10, 49 * (row + 1) + margin_vertical - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return img

# Initialize the environment
frame = initialize_frame()
state, _ = cliffEnv.reset()
done = False

while not done:
    frame2 = put_agent(frame.copy(), state)
    cv2.imshow("Cliff Walking", frame2)
    cv2.waitKey(250)

    action = np.random.randint(0, 4)
    state, reward, terminated, truncated, _ = cliffEnv.step(action)
    done = terminated or truncated

cliffEnv.close()
cv2.destroyAllWindows()
