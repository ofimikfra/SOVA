from collections import Counter

BUFFER_SIZE = 30  # frames to buffer before determining dominant result

# expression buffer
expr_buffer = []
expr_frame_count = 0

# gesture buffer
gest_buffer = []
gest_frame_count = 0

# body action buffer
action_buffer = []
action_frame_count = 0


# expression processing

def processExpression(expression: str, confidence: float = 1.0):
    global expr_frame_count

    expr_buffer.append(expression)  # confidence ignored for now, just buffer the label
    expr_frame_count += 1

    if expr_frame_count >= BUFFER_SIZE:
        dominant = getDominant(expr_buffer, neutral="Neutral")
        expr_buffer.clear()
        expr_frame_count = 0
        return dominant

    return None


# gesture processing

def processGesture(gesture: str):
    """
    Buffer gesture for BUFFER_SIZE frames, then return the dominant one.
    Returns None while still buffering.
    """
    global gest_frame_count

    gest_buffer.append(gesture)
    gest_frame_count += 1

    if gest_frame_count >= BUFFER_SIZE:
        dominant = getDominant(gest_buffer, neutral="No Gesture")
        gest_buffer.clear()
        gest_frame_count = 0
        return dominant

    return None

# body action processing

def processBodyAction(action: str):
    global action_frame_count
    action_buffer.append(action)
    action_frame_count += 1
    if action_frame_count >= BUFFER_SIZE:
        dominant = getDominant(action_buffer, neutral="Person Center")
        action_buffer.clear()
        action_frame_count = 0
        return dominant
    return None


# helper for getting dominant
# Return the most common label in buffer.
# If the neutral label makes up >60% of frames, return the top non-neutral label instead

def getDominant(buffer: list, neutral: str) -> str:
    
    if not buffer:
        return neutral

    counts = Counter(buffer)
    most_common = counts.most_common()
    total = len(buffer)

    if most_common[0][0] == neutral and most_common[0][1] / total > 0.6:
        non_neutral = [(label, n) for label, n in most_common if label != neutral]
        if non_neutral:
            return non_neutral[0][0]

    return most_common[0][0]