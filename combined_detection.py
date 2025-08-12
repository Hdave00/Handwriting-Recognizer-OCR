import numpy as np
import pygame
import sys
import tensorflow as tf
import time
import string

""" This is where we are doing the pygame visualization of how the Neural Network detects handwriting for both letters and, numbers.
    - User can switch between digit mode and letter mode to more accurately let the NN detect input.
    - It can't really be combined in one mode as the MNIST and EMNIST datasets have different collection paramters and are presented differently when exposed
        to the neural network. Especially when using this methodology of visualizing machine learning.    
"""

# Combined label mapping
labels = list(string.digits + string.ascii_uppercase)

# Check command-line arguments
if len(sys.argv) != 2:
    sys.exit("Usage: python detection.py models/model.keras")
model = tf.keras.models.load_model(sys.argv[1])

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Start pygame
pygame.init()
size = width, height = 800, 500  # Changes in size must account for for additional UI changes (if any)
screen = pygame.display.set_mode(size)

# Fonts
OPEN_SANS = "assets/fonts/OpenSans-Regular.ttf"
smallFont = pygame.font.Font(OPEN_SANS, 20)
mediumFont = pygame.font.Font(OPEN_SANS, 30)
largeFont = pygame.font.Font(OPEN_SANS, 40)

# Chose this because to makes sense from the reference code
ROWS, COLS = 28, 28
OFFSET = 20
CELL_SIZE = 10

# Handwriting can be defined as a list for each 0 index * columns for each index in each row's length/size. Ie, we want the handwriting to only be what we
# draw in the grid to be valid intially and that is EMPTY or, all cells are white and not shades meaning its a clean slate.
handwriting = [[0] * COLS for _ in range(ROWS)]
classification = None
prediction_time = None
confidence = None

# Toggle between letter and digit preprocessing
is_letter_mode = True

while True:

    # within the while loop we want the pygame window and all other things below to be active, so that the pygame window stays open.
    # so for every event in the pygame window, if the event is "QUIT" then close program and pygame window 
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

        # elif any key is pressed, and also if the space key is pressed then launch in digit mode
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                is_letter_mode = not is_letter_mode

        # if left click, then record mouse position,
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse = event.pos

            # if the cursor is over the reset button, then clear the handwriting, classification and confidence (hide them)
            if resetButton.collidepoint(mouse):
                handwriting = [[0] * COLS for _ in range(ROWS)]
                classification = None
                confidence = None

            # if mode button pressed, then go in default mode ie, digit
            elif modeButton.collidepoint(mouse):
                is_letter_mode = not is_letter_mode

            # if classify button is pressed then convert input data (cells depressed) into a numpy array similar to what EMNIST expects
            elif classifyButton.collidepoint(mouse):
                input_data = np.array(handwriting).reshape(1, 28, 28, 1)
                
                # CRITICAL: Apply inverse transform for digits
                # if its not a in letter mode then,
                if not is_letter_mode:

                    # For digits, we need to rotate/flip to match EMNIST orientation
                    input_data = np.transpose(input_data, (0, 2, 1, 3))  # Rotate
                    input_data = np.flip(input_data, axis=2)              # Flip
                
                # this is the time taken, prediction made, classification, confidence and time taken capture block
                start = time.time()
                prediction = model.predict(input_data, verbose=0)
                classification = prediction.argmax()
                confidence = prediction.max()
                prediction_time = (time.time() - start) * 1000      # * 1000 because its recorded in ms

    # Continuous mouse drawing handling
    click, _, _ = pygame.mouse.get_pressed()
    mouse_pos = pygame.mouse.get_pos() if click == 1 else None

    screen.fill(BLACK)

    # Draw each grid cell (so for each row and column)
    for i in range(ROWS):
        for j in range(COLS):
            rect = pygame.Rect(
                OFFSET + j * CELL_SIZE,
                OFFSET + i * CELL_SIZE,
                CELL_SIZE, CELL_SIZE
            )

            # If cell has been written on, darken cell
            if handwriting[i][j]:
                channel = 255 - (handwriting[i][j] * 255)
                pygame.draw.rect(screen, (channel, channel, channel), rect)
            else:
                pygame.draw.rect(screen, WHITE, rect)
            pygame.draw.rect(screen, BLACK, rect, 1)

            # If writing on this cell, fill in current cell and neighbors, and normalize it 
            if mouse_pos and rect.collidepoint(mouse_pos):
                handwriting[i][j] = 250 / 255
                if i + 1 < ROWS:
                    handwriting[i + 1][j] = 220 / 255
                if j + 1 < COLS:
                    handwriting[i][j + 1] = 220 / 255
                if i + 1 < ROWS and j + 1 < COLS:
                    handwriting[i + 1][j + 1] = 190 / 255

    # -----  NOTE  -----
    # This button creation block is directly linked to the mode toggle after it. Change carefully.

    # Mode toggle button, what text should the mode button dispkay, and other characteristics of the buttons like size, colour, centering etc
    modeButton = pygame.Rect(
        270, OFFSET + ROWS * CELL_SIZE + 30,
        150, 30
    )

    modeText = smallFont.render(
        f"Mode: {'Letters' if is_letter_mode else 'Digits'}", 
        True, BLACK
    )
    modeTextRect = modeText.get_rect()
    modeTextRect.center = modeButton.center
    pygame.draw.rect(screen, WHITE, modeButton)
    screen.blit(modeText, modeTextRect)

    # Reset button
    resetButton = pygame.Rect(
        30, OFFSET + ROWS * CELL_SIZE + 30,
        100, 30
    )
    resetText = smallFont.render("Reset", True, BLACK)
    resetTextRect = resetText.get_rect()
    resetTextRect.center = resetButton.center
    pygame.draw.rect(screen, WHITE, resetButton)
    screen.blit(resetText, resetTextRect)

    # Classify button
    classifyButton = pygame.Rect(
        150, OFFSET + ROWS * CELL_SIZE + 30,
        100, 30
    )
    classifyText = smallFont.render("Classify", True, BLACK)
    classifyTextRect = classifyText.get_rect()
    classifyTextRect.center = classifyButton.center
    pygame.draw.rect(screen, WHITE, classifyButton)
    screen.blit(classifyText, classifyTextRect)

    # Reset drawing
    if mouse_pos and resetButton.collidepoint(mouse_pos):
        handwriting = [[0] * COLS for _ in range(ROWS)]
        classification = None
        confidence = None

    # -----  NOTE  -----
    """
        - This section for the mode toggle between digits (MNIST) and letters (EMNIST) is EXTREMELY important and, because of the way the data is collected
            and represented, it is very important to note that changes made to this section will reflect directly on the prediction quality even if the
            Convolutional Neural Network is well trained and tested.

        - So do not try to unify this into one pygame grid or logic as it will just make totally bad predictions.
    """

    # Toggle mode
    if mouse_pos and modeButton.collidepoint(mouse_pos):
        is_letter_mode = not is_letter_mode

    # Generate classification when the classify button is pressed, and use that same numpy array logic from before to normalize it according to MNIST format
    if mouse_pos and classifyButton.collidepoint(mouse_pos):
        input_data = np.array(handwriting).reshape(1, 28, 28, 1)
        
        # Apply EMNIST preprocessing only in letter mode, otherwise it'll just use standard digit MNIST config from before
        if is_letter_mode:
            input_data = np.transpose(input_data, (0, 2, 1, 3))  # Rotate
            input_data = np.flip(input_data, axis=2)              # Flip
        
        start = time.time()
        prediction = model.predict(input_data)
        classification = prediction.argmax()
        confidence = prediction.max()
        prediction_time = (time.time() - start) * 1000


        # -----  NOTE  -----
        """
        --- Important note about model ---
            Make sure model is trained to output:

            - 0-9 for digits
            - 10-35 for letters (A-Z)
            - If model was trained differently (eg, letters as 0-25), we need to adjust the mapping accordingly
        """

    # Show classification if one exists 
    if classification is not None:

        # the size of grid is not set at global level (as a unique variable) so we alter that here
        # we want the right side panel to be 2x the size, ie, the info panel in black grid to be 
            # half of the total grid_size + (total width - grid_size) / by 2
        grid_size = OFFSET * 2 + CELL_SIZE * COLS
        right_panel_x = grid_size + ((width - grid_size) / 2)
        
        # Display current mode
        modeDisplay = mediumFont.render(
            f"Mode: {'Letters' if is_letter_mode else 'Digits'}",
            True, WHITE
        )
        screen.blit(modeDisplay, (right_panel_x - modeDisplay.get_width()/2, 50))
        
        # Get correct label based on model output
        if is_letter_mode:

            # For letters, the model outputs 0-25 (A-Z), but lets NOT hardcode that cause its not the best idea obvs
            display_char = chr(ord('A') + classification)  # Convert 0-25 to A-Z
        else:
            # For digits, model outputs 0-9
            display_char = chr(ord('0') + classification) if classification < 10 else '?'
        
        # Display classification
        classificationText = largeFont.render(display_char, True, WHITE)
        screen.blit(classificationText, 
                (right_panel_x - classificationText.get_width()/2, 100))
        
        # Display confidence
        if confidence is not None:
            confText = mediumFont.render(f"Confidence: {confidence*100:.1f}%", True, WHITE)
            screen.blit(confText, (right_panel_x - confText.get_width()/2, 160))
        
        # Display time taken
        if prediction_time is not None:
            timeText = smallFont.render(f"Time: {prediction_time:.2f} ms", True, WHITE)
            screen.blit(timeText, (right_panel_x - timeText.get_width()/2, 200))
        
        # Display instructions
        instrText = smallFont.render("Press SPACE to toggle mode", True, WHITE)
        screen.blit(instrText, (right_panel_x - instrText.get_width()/2, 250))

    pygame.display.flip()