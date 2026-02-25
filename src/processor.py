from collections import Counter

BUFFER_SIZE = 30  # no. frames to buffer before determining dominant expression

expression_buffer = []  # detected expressions within buffer
frame_count = 0  # frames since last reset


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

    # if neutral makes up more than 60% of the buffer, suppress
    total = len(expression_buffer)
    most_common = counts.most_common()

    # return top non-neutral expression
    if most_common[0][0] == "Neutral" and most_common[0][1] / total > 0.6:
        non_neutral = [(expr, n) for expr, n in most_common if expr != "Neutral"] 
        if non_neutral:
            return non_neutral[0][0]

    return most_common[0][0]


'''
TODO: add feature to only return expression during expression changes
        (return expression if expression is different from previous expression)
'''