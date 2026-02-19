from collections import Counter

BUFFER_SIZE = 30  # number of frames to buffer before determining dominant expression

expression_buffer = []  # rolling buffer of detected expressions
frame_count = 0  # tracks frames since last flush


def processExpression(expression):
    global frame_count

    expression_buffer.append(expression)
    frame_count += 1

    if frame_count >= BUFFER_SIZE:
        dominant = getDominantExpression()
        expression_buffer.clear()
        frame_count = 0
        return dominant

    return None  # still buffering


def getDominantExpression():
    if not expression_buffer:
        return "Neutral"

    counts = Counter(expression_buffer)

    # if neutral makes up more than 60% of the buffer, suppress so a brief-but-real expression isn't drowned out
    total = len(expression_buffer)
    most_common = counts.most_common()

    if most_common[0][0] == "Neutral" and most_common[0][1] / total > 0.6:
        # return top non-neutral expression if one exists
        non_neutral = [(expr, n) for expr, n in most_common if expr != "Neutral"]
        if non_neutral:
            return non_neutral[0][0]

    return most_common[0][0]